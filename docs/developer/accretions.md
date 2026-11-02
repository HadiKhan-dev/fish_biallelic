# Retained accretions and cost exemptions

An accretion is a retained special rule, bounded-search device or diagnostic whose purpose merits separate review. Its presence is explicit; this register does not authorise changing it. “Review for a later revision” is a review status, not a promised removal. Dead units are excluded from methods.

## Cost exemptions

| Decision | Retained scope | Cost/limitation |
|---|---|---|
| 0 | Rare context-coded selection screening | O(K⁵) per round; at K = 30 the K-only factor is 24,300,000 |
| 0b | Ranked assembly/refinement seed refits | Fixed ranked_refits admission can omit candidates at larger K; K = 30 does not expand the cap |
| 0c | Bounded discovery and selection searches | Discovery O(N L K⁴); selection backstop O(N L K³(K+L)) and context-snap passes O((K+L)²); K⁴ = 810,000 at K = 30 |
| 0d | Optional pairwise IBD | Off by default; O(N² L), with 3,123,750 unordered pairs at N = 2,500 |

These are algebraic factors, not measured operation counts or timings. See [costs](costs.md) for their scope. Four further worst-case costs above cubic in K (assembly joins' structural screening, refinement window fixing, painting label-continuity checks and founder-refit ties) are carried exactly as documented exceptions under owner decision 0e, the treatment of 0c; the [cost page](costs.md#exemptions-and-exceptions) names them. They are listed for review in release 4.1.

## Accretion register

| Area / unit | Rule | Current and later-review status |
|---|---|---|
| assembly-joins / [AJ10](../methods/assembly.md#aj10) | Constant-transition Cartesian family (first-covering proposals) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| assembly-joins / [AJ25](../methods/assembly.md#aj25) | Backstops (sweep/count/pool) and final death-test census | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| assembly-joins / [AJ35](../methods/assembly.md#aj35) | Shadow-envelope proof certificate (verification metadata) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| discovery / [DI07](../methods/discovery.md#di07) | Fit memo, frontier batching and fixed-point certificates | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| discovery / [DI29](../methods/discovery.md#di29) | Convention-reversal switch and diagnostics | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| engine / [ENG-12](../methods/execution.md#eng-12) | Environment switches (hidden state) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| engine / [ENG-28](../methods/execution.md#eng-28) | Kernel manifest and compile audit | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| engine / [ENG-08](../methods/execution.md#eng-08) (part) | Completion markers | Carried as workflow markers for external workflow tools, written after the cut checks its products; a marker never authorises reuse. |
| engine / [ENG-09](../methods/execution.md#eng-09) (part) | Removed-section handling | Carried: removed metadata sections and the old preset are refused with a reason by the loader and `migrate-config`. |
| engine / [ENG-18](../methods/execution.md#eng-18) (part) | Layout guard | Carried: the store rejects foreign or earlier checkpoint layouts before any publication. |
| engine / [ENG-25](../methods/execution.md#eng-25) (part) | Byte-reservation interface | Not carried: 4.0 has no byte-reservation interface; working-set estimates are diagnostics and admission follows measured memory pressure. |
| engine / ENG-29 | Batch queue (cross-allocation helpers) | Removed helper queue; workflow cuts supply orchestration. No retained methods unit. |
| evidence-core / [EC02](../methods/evidence.md#ec02) | Observation-source identity (VCF path, size, mtime_ns, header order; simulation content identity) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| evidence-core / [EC34](../methods/core.md#ec34) | Valley census (HAPLOTYPES_TIE_CENSUS) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| local-selection / [LS13](../methods/selection.md#ls13) | Whole-search reuse and block-scoped memoisation | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| local-selection / [LS20](../methods/selection.md#ls20) | Bank budgets: start backstop, family admission at the score cap, round and score backstops | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| local-selection / [LS29](../methods/selection.md#ls29) | Recorded proposal summaries (per-bin background counts, combined posterior and support) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| local-selection / [LS30](../methods/selection.md#ls30) | Round execution, scheduling, records, identities and resume | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| local-selection / [LS31](../methods/selection.md#ls31) | Diagnostics: tie counters, valley census, summaries and reports | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU06](../methods/export.md#ou06) | Founder identity flags (duplicate-partner changes) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU07](../methods/export.md#ou07) | Founder track TSVs (tracks product) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU13](../methods/export.md#ou13) | Painting and assembly diagnostic tables | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU17](../methods/export.md#ou17) | Stage summary tables (founder_refit, family_phase) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU19](../methods/export.md#ou19) | Ties and ambiguity report | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU20](../methods/simulation.md#ou20) | Template builder (windowed discovery on fixed-error evidence) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU21](../methods/simulation.md#ou21) | Naive window linker | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| outputs / [OU22](../methods/simulation.md#ou22) | Synthetic example generator | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| painting-refit / [PR14](../methods/painting.md#pr14) | Decoder execution: memory-granted sample batches and the packed-emission cache | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| pedigree-decision / [PD07](../methods/pedigree.md#pd07) | Junction-density order probability (graph visit order, role probability) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| pedigree-decision / [PD26](../methods/pedigree.md#pd26) | Relative-proxy contrasts | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| pedigree-scoring / [PS13](../methods/pedigree.md#ps13) | Robust screen utilities and search diagnostics | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| phase-recombination / [FP22](../methods/phase.md#fp22) | Storage routes and release scheduling (resident/windowed, serial/pooled, spool, mapped outputs) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| phase-recombination / [FP24](../methods/phase.md#fp24) | Diagnostics, summary and legacy fields (coverage, zero frames) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| phase-recombination / [FP25](../methods/phase.md#fp25) | Full-objective delta-check reference path (check_deltas) | Carried as a test-only reference wrapper; no production mode switch. |
| refinement-count / [RC04](../methods/refinement.md#rc04) | Evidence windows: resident or streamed route (3.4), leases, cover, exact fallback | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| refinement-count / [RC07](../methods/refinement.md#rc07) | SW5 message storage route: dense or 16-block checkpoints; reconstruction reader and view lifetimes | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| refinement-count / [RC14](../methods/refinement.md#rc14) | Zero-gain cleanup of unused founders | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| refinement-count / [RC20](../methods/refinement.md#rc20) | Level orchestration: scheduling, worker pools, memory admission | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| refinement-search / [RS06](../methods/refinement.md#rs06) | Window-programme frontier cap and memory-budget retry schedule | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| refinement-search / [RS07](../methods/refinement.md#rs07) | Per-template window work account and whole-component stop | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| refinement-search / [RS18](../methods/refinement.md#rs18) | Weak label junctions (diagnostic) | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| relatedness / [RL04](../methods/relatedness.md#rl04) | Pairwise three-state IBD chain and its genome-wide summaries | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |
| relatedness / [RL18](../methods/relatedness.md#rl18) | Final half-sib links | Carried unchanged except declared ledger normalisations; review purpose/cost before a later scientific revision. |

Numerical limits in the inventory labels identify the originating rules; their current values and provenance are authoritative only in the [generated parameter reference](../reference/parameters.md). The [migration guide](../user/migration.md) records removed interfaces.
