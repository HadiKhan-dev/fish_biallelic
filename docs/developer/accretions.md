# Retained accretions and cost exemptions

An accretion is a retained special rule, bounded-search device or diagnostic whose purpose merits separate review. Its presence is explicit; this register does not authorise changing it. The register describes current mechanisms; changing a scientific rule requires review.

## Cost exemptions

| Decision | Retained scope | Cost/limitation |
|---|---|---|
| 0 | Context-coded selection screening | O(K⁴) per round from certificates and start de-duplication, plus at most 128 cubic exact-code solves; at K = 30 the quartic K-only factor is 810,000 |
| 0c | Bounded discovery and selection searches | Discovery O(N L K⁴); selection backstop O(N L K³(K+L)) and context-snap passes O((K+L)²); K⁴ = 810,000 at K = 30. discovery's replacement starts (48/96) and archive beam (6/12), and selection's backstops, bank budgets, context snaps and LS10/LS17 caps bind at every K; they are bounded-search devices that need new search or stopping contracts, not larger constants |

These are algebraic factors, not measured operation counts or timings. See [costs](costs.md) for their scope. Three further worst-case costs above cubic in K (assembly joins' structural screening, refinement window fixing and founder-refit ties) are carried exactly as documented exceptions under owner decision 0e, the treatment of 0c; the [cost page](costs.md#exemptions-and-exceptions) names them with their current scope.

## Accretion register

| Area / unit | Rule | Current scope |
|---|---|---|
| assembly-joins / [AJ10](../methods/assembly.md#aj10) | Constant-transition Cartesian family (first-covering proposals) | a proposal rule (first-covering subset of the constant-linkage family), not a cap. |
| assembly-joins / [AJ25](../methods/assembly.md#aj25) | Backstops (sweep/count/pool) and final death-test census | Sweep, count and pool guards bound search; a binding pool guard reports certified = false. The final death census is diagnostic. |
| discovery / [DI07](../methods/discovery.md#di07) | Fit memo, frontier batching and fixed-point certificates | exact memos, frontier batching and fixed-point certificates decide nothing a fresh fit would not. |
| discovery / [DI29](../methods/discovery.md#di29) | Convention-reversal switch and diagnostics | the reversal switch is the tie-convention control, its diagnostics are compact. |
| engine / [ENG-12](../methods/execution.md#eng-12) | Environment switches (hidden state) | ambient NUMBA_* code-generation inputs key the kernel cache but are not part of the numerical policy (environment reference lists them). |
| engine / [ENG-28](../methods/execution.md#eng-28) | Kernel manifest and compile audit | compile manifest and audit decide no scientific value. |
| engine / [ENG-08](../methods/execution.md#eng-08) (part) | Completion markers | Carried as workflow markers for external workflow tools, written after the cut checks its products; a marker never authorises reuse. |
| engine / [ENG-09](../methods/execution.md#eng-09) (part) | Removed-section handling | Carried: removed metadata sections and the old preset are refused with a reason by the loader. |
| engine / [ENG-18](../methods/execution.md#eng-18) (part) | Layout guard | Carried: the store rejects foreign or earlier checkpoint layouts before any publication. |
| engine / [ENG-25](../methods/execution.md#eng-25) (part) | Byte-reservation interface | working-set estimates are diagnostics and admission follows measured memory pressure. |
| evidence-core / [EC02](../methods/evidence.md#ec02) | Observation-source identity (VCF path, size, mtime_ns, header order; simulation content identity) | the consumed-source digest protects evidence reuse; the path/stat tuple only keys the digest cache. |
| evidence-core / [EC34](../methods/core.md#ec34) | Valley census (explicit diagnostic sink) | optional valley census through an explicit sink, no decision feedback. |
| local-selection / [LS13](../methods/selection.md#ls13) | Whole-search reuse and block-scoped memoisation | Block-scoped memoisation reuses a fit only under its declared dependency predicate. |
| local-selection / [LS20](../methods/selection.md#ls20) | Bank budgets: start backstop, family admission at the score cap, round and score backstops | Bank, round and score budgets bound search and record omissions when they bind. |
| local-selection / [LS29](../methods/selection.md#ls29) | Recorded proposal summaries (per-bin background counts, combined posterior and support) | computed for every covered component and read by no selection decision; kept as published report content. |
| local-selection / [LS30](../methods/selection.md#ls30) | Round execution, scheduling, records, identities and resume | execution, records and resume only. |
| local-selection / [LS31](../methods/selection.md#ls31) | Diagnostics: tie counters, valley census, summaries and reports | compact counters, census opt-in. |
| outputs / [OU06](../methods/export.md#ou06) | Founder identity flags (duplicate-partner changes) | owner lookup and inverse row maps computed once; same predicate, fields and order. |
| outputs / [OU07](../methods/export.md#ou07) | Founder track TSVs (tracks product) | direct formatting of released alleles. |
| outputs / [OU13](../methods/export.md#ou13) | Painting and assembly diagnostic tables | support statistics formatted by one helper; same tables, values and text. |
| outputs / [OU17](../methods/export.md#ou17) | Stage summary tables (founder_refit, family_phase) | readers project recorded diagnostics. |
| outputs / [OU19](../methods/export.md#ou19) | Ties and ambiguity report | typed projection after the science, no feedback. |
| outputs / [OU22](../methods/simulation.md#ou22) | Synthetic example generator | seeded onboarding generator outside production inference. |
| painting-refit / [PR14](../methods/painting.md#pr14) | Decoder execution: memory-granted sample batches and the packed-emission cache | kept as execution organisation (every sample and bin is processed); batch sizing does not select scientific work. |
| pedigree-decision / [PD07](../methods/pedigree.md#pd07) | Junction-density order probability (graph visit order, role probability) | a scientific rule (graph visit order decides conflicting cycles); any change needs its own review. |
| pedigree-decision / [PD26](../methods/pedigree.md#pd26) | Relative-proxy contrasts | report-only contrasts; no parent, count or release decision reads them. |
| pedigree-scoring / [PS13](../methods/pedigree.md#ps13) | Robust screen utilities and search diagnostics | screen utilities and diagnostics feed search_diagnostics only; no panel membership or call reads them. |
| phase-recombination / [FP22](../methods/phase.md#fp22) | Storage routes and release scheduling (resident/windowed, serial/pooled, spool, mapped outputs) | both storage and both execution routes kept; release scheduling is explicit. |
| phase-recombination / [FP24](../methods/phase.md#fp24) | Diagnostics, summary and legacy fields (coverage, zero frames) | diagnostics also supply recombination's convergence report without deciding events. |
| phase-recombination / [FP25](../methods/phase.md#fp25) | Full-objective enumeration and threshold reference checks | The model evaluates the full objective for ascent history and uses local row differences for proposals. |
| refinement-count / [RC04](../methods/refinement.md#rc04) | Evidence windows: resident or streamed route, leases, cover, exact fallback | execution route with exact fallback, no candidate filter. |
| refinement-count / [RC07](../methods/refinement.md#rc07) | SW5 message storage route: dense or 16-block checkpoints; reconstruction reader and view lifetimes | exact recurrences in both storage routes. |
| refinement-count / [RC20](../methods/refinement.md#rc20) | Level orchestration: scheduling, worker pools, memory admission | ordered orchestration and memory admission. |
| refinement-count / [RC22](../methods/refinement.md#rc22) | Component window store, shared polish answers and concurrent count histories | exact reuse of identical window and polish problems and concurrent independent count histories; outcomes are the sequential driver's. |
| refinement-search / [RS06](../methods/refinement.md#rs06) | Shared messages and bounded search storage | Shared max-plus messages reuse unchanged path segments; storage admission follows actual problem dimensions and the fixed frontier. |
| refinement-search / [RS07](../methods/refinement.md#rs07) | Per-template window work account and whole-component stop | The per-template work account stops an unfinished component and reports its domain. |
| refinement-search / [RS18](../methods/refinement.md#rs18) | Weak label junctions (diagnostic) | reported intervals only. |
| refinement-search / [RS19](../methods/refinement.md#rs19) | Exact reuse inside a fixed-K search (restated windows, relabelled messages, shared sweep 0, standing-offer gains) | exact reuse of renamed or unchanged problems; decisions, records and logical work accounts are those of computing afresh. |
| relatedness / [RL18](../methods/relatedness.md#rl18) | Final half-sib links | released half-sib links; no feedback to fitted families or the pedigree; a scientific rule. |

Numerical limits in the inventory labels identify the originating rules; their current values and provenance are authoritative only in the [generated parameter reference](../reference/parameters.md).
The count rule admits its natural candidate domain. The accepted cost is cubic per count round and quartic across deletion rounds; see [costs](costs.md#natural-domain-count-refits).
