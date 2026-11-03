# Retained accretions and cost exemptions

An accretion is a retained special rule, bounded-search device or diagnostic whose purpose merits separate review. Its presence is explicit; this register does not authorise changing it. “Review for a later revision” is a review status, not a promised removal. Dead units are excluded from methods.

## Cost exemptions

| Decision | Retained scope | Cost/limitation |
|---|---|---|
| 0 | Rare context-coded selection screening | O(K⁵) per round; at K = 30 the K-only factor is 24,300,000 |
| 0c | Bounded discovery and selection searches | Discovery O(N L K⁴); selection backstop O(N L K³(K+L)) and context-snap passes O((K+L)²); K⁴ = 810,000 at K = 30. Reviewed 4.2: discovery's replacement starts (48/96) and archive beam (6/12), and selection's backstops, bank budgets, context snaps and LS10/LS17 caps bind at every K; they are bounded-search devices that need new search or stopping contracts, not larger constants |
| 0d | Optional pairwise IBD | Off by default; O(N² L), with 3,123,750 unordered pairs at N = 2,500 |

These are algebraic factors, not measured operation counts or timings. See [costs](costs.md) for their scope. Four further worst-case costs above cubic in K (assembly joins' structural screening, refinement window fixing, painting label-continuity checks and founder-refit ties) are carried exactly as documented exceptions under owner decision 0e, the treatment of 0c; the [cost page](costs.md#exemptions-and-exceptions) names them with their 4.2 review.

## Accretion register

| Area / unit | Rule | Current and later-review status |
|---|---|---|
| assembly-joins / [AJ10](../methods/assembly.md#aj10) | Constant-transition Cartesian family (first-covering proposals) | Reviewed 4.2: kept; a proposal rule (first-covering subset of the constant-linkage family), not a cap. |
| assembly-joins / [AJ25](../methods/assembly.md#aj25) | Backstops (sweep/count/pool) and final death-test census | Reviewed 4.2: sweep and count guards kept (no hit in 122,504 stored joins); the 1,024-endpoint pool guard stops 148 stored joins (std, backcross, TroMau) with certified = false, and a continuation design is open; the final death census stays diagnostic. |
| assembly-joins / [AJ35](../methods/assembly.md#aj35) | Shadow-envelope proof certificate (verification metadata) | Reviewed 4.2: kept; verification metadata only, never a count or allele choice. |
| discovery / [DI07](../methods/discovery.md#di07) | Fit memo, frontier batching and fixed-point certificates | Reviewed 4.2: kept; exact memos, frontier batching and fixed-point certificates decide nothing a fresh fit would not. |
| discovery / [DI29](../methods/discovery.md#di29) | Convention-reversal switch and diagnostics | Reviewed 4.2: kept; the reversal switch is the tie-convention control, its diagnostics are compact. |
| engine / [ENG-12](../methods/execution.md#eng-12) | Environment switches (hidden state) | Reviewed 4.2: the historical science switches are retired; ambient NUMBA_* code-generation inputs key the kernel cache but are not part of the numerical policy (environment reference lists them). |
| engine / [ENG-28](../methods/execution.md#eng-28) | Kernel manifest and compile audit | Reviewed 4.2: kept; compile manifest and audit decide no scientific value. |
| engine / [ENG-08](../methods/execution.md#eng-08) (part) | Completion markers | Carried as workflow markers for external workflow tools, written after the cut checks its products; a marker never authorises reuse. |
| engine / [ENG-09](../methods/execution.md#eng-09) (part) | Removed-section handling | Carried: removed metadata sections and the old preset are refused with a reason by the loader and `migrate-config`. |
| engine / [ENG-18](../methods/execution.md#eng-18) (part) | Layout guard | Carried: the store rejects foreign or earlier checkpoint layouts before any publication. |
| engine / [ENG-25](../methods/execution.md#eng-25) (part) | Byte-reservation interface | Not carried: 4.0 has no byte-reservation interface; working-set estimates are diagnostics and admission follows measured memory pressure. |
| engine / ENG-29 | Batch queue (cross-allocation helpers) | Removed helper queue; workflow cuts supply orchestration. No retained methods unit. |
| evidence-core / [EC02](../methods/evidence.md#ec02) | Observation-source identity (VCF path, size, mtime_ns, header order; simulation content identity) | Reviewed 4.2: kept; the consumed-source digest protects evidence reuse; the path/stat tuple only keys the digest cache. |
| evidence-core / [EC34](../methods/core.md#ec34) | Valley census (HAPLOTYPES_TIE_CENSUS) | Reviewed 4.2: kept; optional valley census through an explicit sink, no decision feedback. |
| local-selection / [LS13](../methods/selection.md#ls13) | Whole-search reuse and block-scoped memoisation | Reviewed 4.2: block-scoped memoisation kept; whole-search reuse (1,509 std and 812 TroMau calls) needs a dependency-complete reuse predicate, a design item. |
| local-selection / [LS20](../methods/selection.md#ls20) | Bank budgets: start backstop, family admission at the score cap, round and score backstops | Reviewed 4.2: binds (K = 30: 113 of 400 context calls; TroMau 24 of 19,974); needs explicit omissions and certified stopping rather than larger coefficients, a design item. |
| local-selection / [LS29](../methods/selection.md#ls29) | Recorded proposal summaries (per-bin background counts, combined posterior and support) | Reviewed 4.2: computed for every covered component and read by no selection decision; kept as published report content. |
| local-selection / [LS30](../methods/selection.md#ls30) | Round execution, scheduling, records, identities and resume | Reviewed 4.2: kept; execution, records and resume only. |
| local-selection / [LS31](../methods/selection.md#ls31) | Diagnostics: tie counters, valley census, summaries and reports | Reviewed 4.2: kept; compact counters, census opt-in. |
| outputs / [OU06](../methods/export.md#ou06) | Founder identity flags (duplicate-partner changes) | Reviewed 4.2: owner lookup and inverse row maps computed once; same predicate, fields and order. |
| outputs / [OU07](../methods/export.md#ou07) | Founder track TSVs (tracks product) | Reviewed 4.2: kept; direct formatting of released alleles. |
| outputs / [OU13](../methods/export.md#ou13) | Painting and assembly diagnostic tables | Reviewed 4.2: support statistics formatted by one helper; same tables, values and text. |
| outputs / [OU17](../methods/export.md#ou17) | Stage summary tables (founder_refit, family_phase) | Reviewed 4.2: kept; readers project recorded diagnostics. |
| outputs / [OU19](../methods/export.md#ou19) | Ties and ambiguity report | Reviewed 4.2: kept; typed projection after the science, no feedback. |
| outputs / [OU20](../methods/simulation.md#ou20) | Template builder (windowed discovery on fixed-error evidence) | Reviewed 4.2: a scientific generating assumption of the simulation templates; changing it changes simulation truth. |
| outputs / [OU21](../methods/simulation.md#ou21) | Naive window linker | Reviewed 4.2: a scientific generating assumption of the simulation templates (path and overlap rules). |
| outputs / [OU22](../methods/simulation.md#ou22) | Synthetic example generator | Reviewed 4.2: kept; seeded onboarding generator outside production inference. |
| painting-refit / [PR14](../methods/painting.md#pr14) | Decoder execution: memory-granted sample batches and the packed-emission cache | Reviewed 4.2: kept as execution organisation (every sample and bin is processed); its unused minimum-memory setting is removed. |
| pedigree-decision / [PD07](../methods/pedigree.md#pd07) | Junction-density order probability (graph visit order, role probability) | Reviewed 4.2: a scientific rule (graph visit order decides conflicting cycles); any change needs its own review. |
| pedigree-decision / [PD26](../methods/pedigree.md#pd26) | Relative-proxy contrasts | Reviewed 4.2: report-only contrasts; no parent, count or release decision reads them. |
| pedigree-scoring / [PS13](../methods/pedigree.md#ps13) | Robust screen utilities and search diagnostics | Reviewed 4.2: screen utilities and diagnostics feed search_diagnostics only; no panel membership or call reads them. |
| phase-recombination / [FP22](../methods/phase.md#fp22) | Storage routes and release scheduling (resident/windowed, serial/pooled, spool, mapped outputs) | Reviewed 4.2: both storage and both execution routes kept; the duplicate CAL2 chunk wrapper is removed. |
| phase-recombination / [FP24](../methods/phase.md#fp24) | Diagnostics, summary and legacy fields (coverage, zero frames) | Reviewed 4.2: kept; diagnostics also supply recombination's convergence report without deciding events. |
| phase-recombination / [FP25](../methods/phase.md#fp25) | Full-objective enumeration and threshold reference checks | Small-objective enumeration and strict gain/keep threshold tests; production uses local row differences and diagnostic tracing. |
| refinement-count / [RC04](../methods/refinement.md#rc04) | Evidence windows: resident or streamed route (3.4), leases, cover, exact fallback | Reviewed 4.2: kept; execution route with exact fallback, no candidate filter. |
| refinement-count / [RC07](../methods/refinement.md#rc07) | SW5 message storage route: dense or 16-block checkpoints; reconstruction reader and view lifetimes | Reviewed 4.2: kept; exact recurrences in both storage routes. |
| refinement-count / [RC20](../methods/refinement.md#rc20) | Level orchestration: scheduling, worker pools, memory admission | Reviewed 4.2: kept; ordered orchestration and memory admission. |
| refinement-search / [RS06](../methods/refinement.md#rs06) | Window-programme frontier cap and memory-budget retry schedule | Reviewed 4.2: kept; no frontier fallback in the 55-case matrix. |
| refinement-search / [RS07](../methods/refinement.md#rs07) | Per-template window work account and whole-component stop | Reviewed 4.2: no work stop in the 55-case matrix; canonical fixing still lacks a construction bound (design item). |
| refinement-search / [RS18](../methods/refinement.md#rs18) | Weak label junctions (diagnostic) | Reviewed 4.2: kept; reported intervals only. |
| relatedness / [RL04](../methods/relatedness.md#rl04) | Pairwise three-state IBD chain and its genome-wide summaries | Reviewed 4.2: optional output (decision 0d); independent of family fit and pedigree. |
| relatedness / [RL18](../methods/relatedness.md#rl18) | Final half-sib links | Reviewed 4.2: released half-sib links; no feedback to fitted families or the pedigree; a scientific rule. |

Numerical limits in the inventory labels identify the originating rules; their current values and provenance are authoritative only in the [generated parameter reference](../reference/parameters.md). The [migration guide](../user/migration.md) records removed interfaces.

The 4.2 count rule admits its natural candidate domain; exemption 0b and the RC14 cleanup accretion are removed. The accepted cost is cubic per count round and quartic in the worst case across deletion rounds; see [costs](costs.md#natural-domain-count-refits).
