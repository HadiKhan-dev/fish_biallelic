# S7. Supplementary tables

| Stage | Probe | What the probe covers | K slope (K = 5–30, N = 200) | N slope (N = 250–2,500, K = 15) |
|-------------|-------------------|--------------------------------------|---------------|---------------|
| evidence | evidence.calibration_objective | sample objective/gradient at 230 depth-alt cells; no intrinsic K | -0.00 ± 0.00 (≤ cubic) | 1.00 ± 0.00 (measured) |
| evidence | evidence.likelihoods | read-model likelihood application N×256; no intrinsic K | -0.00 ± 0.00 (≤ cubic) | 0.98 ± 0.02 (measured) |
| evidence | evidence.quality_genotypes | genotype expectations, default 60 iterations, N×256; no intrinsic K | -0.00 ± 0.00 (≤ cubic) | 1.12 ± 0.01 (measured) |
| relatedness.families | relatedness.screen | all singleton family screen; F=N; no intrinsic K | -0.02 ± 0.01 (≤ cubic) | 2.00 ± 0.00 (measured) |
| relatedness.families | relatedness.terms | AF and family terms; no intrinsic K | -0.01 ± 0.01 (≤ cubic) | 1.02 ± 0.04 (measured) |
| relatedness.pairs | relatedness.pairs | all pairs, 16 thinned loci; no intrinsic K; exemption 0d | 0.03 ± 0.01 (exempt) | 1.99 ± 0.00 (exempt) |
| discovery.fit | discovery.assignment | full fixed-K A step L32, cached evidence patterns; K² per assignment | 1.57 ± 0.08 (≤ cubic) | 0.95 ± 0.03 (measured) |
| discovery.fit | discovery.fit_family | complete two-schedule fit family L32; fresh fit memos; default 50 iterations; exemption 0c | 1.28 ± 0.08 (exempt) | 0.91 ± 0.05 (exempt) |
| discovery.search | discovery.cavity | held-out cavity L32/default 100 iterations; search budget exemption 0c remains analytic | 2.58 ± 0.30 (exempt) | 0.15 ± 0.12 (exempt) |
| discovery.release | discovery.release | fixed-assignment founder release K×32 with N carriers | 0.13 ± 0.01 (≤ cubic) | 1.03 ± 0.03 (measured) |
| selection.fit | selection.exact_pass | EXACT forward/backward pass L32, memo-free; per-pass K2 | 1.62 ± 0.06 (≤ cubic) | 0.99 ± 0.01 (measured) |
| selection.fit | selection.fit_fixed | full fit L32 with unchanged default backstop; exemption 0c; adaptive path | 1.64 ± 0.10 (exempt) | 0.96 ± 0.01 (exempt) |
| selection.search | selection.context_matching | K² context matching kernels of K×K costs per round; exemption 0; no intrinsic N | 4.01 ± 0.05 (exempt) | 0.01 ± 0.01 (exempt) |
| assembly.completion | assembly.cavity_fill | whole-bin cavity fill L64, ten-percent unknown calls, default budgets | 1.62 ± 0.13 (≤ cubic) | 0.96 ± 0.02 (measured) |
| assembly.completion | assembly.crossfit | two-fold crossfit L32, ten-percent unknown calls, default budgets | 1.34 ± 0.08 (≤ cubic) | 0.81 ± 0.03 (measured) |
| assembly.site_model | assembly.site_posterior | weak informative site, all K uncertain; exact/core route recorded | 3.06 ± 0.11 (≤ cubic) | 1.00 ± 0.00 (measured) |
| assembly.linker | assembly.linker_scan | forward micro-HMM L32 with nonuniform priors; no zero-prior memo reuse | 1.90 ± 0.03 (≤ cubic) | 0.98 ± 0.02 (measured) |
| assembly.joins | assembly.join_score | exact panel score, four blocks of four bins; structural-query count remains unresolved | 1.75 ± 0.06 (≤ cubic) | 1.02 ± 0.02 (measured) |
| assembly.shadow_refits | assembly.count_rule | K count decisions with default bootstrap; natural-domain fits and repeated rounds separate | 1.00 ± 0.01 (kernel only) | 1.28 ± 0.08 (kernel only) |
| assembly.orchestration_release | assembly.release | full release of four-block K-row component without alternatives; no intrinsic N | 0.10 ± 0.01 (≤ cubic) | -0.01 ± 0.04 (measured) |
| refinement.search | refinement.score | direct exact whole-site score L64; canonical-fixing complexity remains unresolved | 1.71 ± 0.06 (≤ cubic) | 1.00 ± 0.00 (measured) |
| refinement.count | refinement.count_bound | exact one-deletion upper bounds in four blocks with K+1 rows; natural-domain fits and repeated rounds separate | 1.85 ± 0.12 (kernel only) | 1.00 ± 0.00 (kernel only) |
| painting.coalescence | painting.transposition_gains | all K(K-1)/2 transposition gains at two links; worst candidate rechecks remain analytic | 2.39 ± 0.05 (≤ cubic) | 1.00 ± 0.00 (measured) |
| painting | painting.chain | structured sum-product, N×8 bins, Sprime=2K+1, masses enabled | 1.94 ± 0.01 (≤ cubic) | 1.00 ± 0.00 (measured) |
| painting | painting.decode_masks | both Viterbi conventions plus interval/canonical masks, N×8 bins | 1.73 ± 0.02 (≤ cubic) | 1.02 ± 0.00 (measured) |
| founder_refit | refit.alleles | conditional refit N×32, default three sweeps; random mismatches, not adversarial tie classes | 0.88 ± 0.08 (≤ cubic) | 0.71 ± 0.03 (measured) |
| pedigree.prepare | pedigree.sources | source factors and marginals, N×8 bins×S² | 1.79 ± 0.06 (≤ cubic) | 1.07 ± 0.03 (measured) |
| pedigree.screen_pair_bound | pedigree.pair_bound | all pairs of N-1 candidate parents for one child, two bins, G=S | 2.17 ± 0.16 (≤ cubic) | 2.00 ± 0.00 (measured) |
| pedigree.scores_exposure | pedigree.exposure | all N(N-1)/2 edges plus N trios, two bins, all S(S+1)/2 features | 1.58 ± 0.05 (≤ cubic) | 2.00 ± 0.00 (measured) |
| pedigree.decision | pedigree.family_messages | dense N² M0/M1 alternative family log-odds pass; iterative fit separate; no intrinsic K | 0.00 ± 0.01 (≤ cubic) | 2.01 ± 0.02 (measured) |
| pedigree.resampling_release | pedigree.resample_counts | full configured bootstrap replicate accumulator, fit kernel separate; no intrinsic K | 0.01 ± 0.01 (≤ cubic) | 1.71 ± 0.31 (measured) |
| phase.solve | phase.viterbi | all N root-sample blocks, eight bins, S=K+1; descendant paths separate | 1.93 ± 0.04 (≤ cubic) | 1.01 ± 0.00 (measured) |
| phase.release | phase.posterior | all N conditional sum-product blocks, eight bins, two gauges | 2.81 ± 0.02 (≤ cubic) | 1.00 ± 0.00 (measured) |
| recombination | recombination.follow | N meiosis chain following, 256 bins; no intrinsic K | -0.03 ± 0.02 (≤ cubic) | 1.00 ± 0.00 (measured) |
| sites_export | outputs.samples | actual BCF sample writer N×256; no intrinsic K | 0.01 ± 0.01 (≤ cubic) | 0.90 ± 0.03 (measured) |
| reports | outputs.painting_report | actual founder support tables; all K pooled rows; N×K×256 | 0.90 ± 0.02 (≤ cubic) | 0.97 ± 0.00 (measured) |
| simulation | simulation.reads | independent-site read generator N×256; K founder source only | 0.07 ± 0.01 (≤ cubic) | 0.99 ± 0.01 (measured) |
| evaluation | evaluation.founders | full founder evaluator K rows,F=2K,L=256; no intrinsic N | 1.24 ± 0.11 (≤ cubic) | -0.00 ± 0.01 (measured) |
| contracts | contracts.partners | partner changes B=32; no intrinsic N | 0.86 ± 0.07 (≤ cubic) | 0.01 ± 0.01 (measured) |
| core.ties_numerics | core.ties | workload proxy length=N*K; not a scientific K gate | 1.07 ± 0.03 (size proxy) | 1.10 ± 0.00 (size proxy) |
| kernel_warmup | core.kernel_manifest | signature encode/decode proxy count=N*K; cold compilation separate | 1.00 ± 0.00 (size proxy) | 1.00 ± 0.00 (size proxy) |
| store | store.codec | payload encode/decode proxy [N,K,32]; no scientific K gate | 0.90 ± 0.03 (size proxy) | 1.03 ± 0.02 (size proxy) |
| engine_graph | engine.graph | independent-task validation proxy count=N*K; not scheduler throughput | 1.98 ± 0.01 (size proxy) | 1.99 ± 0.00 (size proxy) |
| interface | interface.settings | settings resolution proxy: TOML load, merge, typed build, validation and settings_text of a simulation configuration with cohorts (K, N) negligible; no intrinsic N or K | 0.00 ± 0.00 (size proxy) | -0.00 ± 0.01 (size proxy) |

Table: Kernel ladders for the method in release 4.4.0. Each probe times one kernel on direct synthetic inputs, on one thread, after a discarded first call; slopes are fitted log–log exponents with standard errors. A K slope at or below 3.3 is accepted as at most cubic; kernel-only rows do not bound complete count-refit passes; “exempt” marks a stage under a decided cost exemption, and “size proxy” an operational kernel whose size stands in for K and N. Source: docs/developer/costs.md.

| Entry | Category | Form | Where it acts |
|-------|-----------------|----------------------------------------------|------------------------------|
| L1 | Formula unification | One Haldane recombination fraction, computed with `expm1`, for every map distance | genetic map, phase edge rates, pedigree selector switches |
| L2 | Summation order | One declared order (sample, natural or C order) for sums whose order otherwise followed worker counts, memory-sized batches or array memory order; the painting estimator's parameters are compared on their 2^-6 log grid | painting estimator, phase release, phase storage routes, pedigree inputs, genome likelihood, junction schedules |
| L3 | Tie order | A declared, stable content order decides exchanges between exact duplicates | assembly ties, refinement ties |
| L4 | Dtype at a test | Family phase's chain test uses float64; float32 only for storage | phase posterior and release |
| L5 | Deterministic replacement of a memory-dependent choice | The shadow-domain proof is attempted by a dimension rule, never by available memory | shadow-envelope proof, refinement count |
| L6 | Label numbering | Family numbers are canonical dense ranks ordered by each family's smallest member | relatedness release |
| L7 | Deterministic replacement of an object-identity choice | A stall's second count step runs only when the extension pool's content differs; `source_decision` names follow the decision sequence | join records, count steps |
| L8 | Environment flags | Tie reversal and the census are explicit settings passed through typed boundaries | tie settings, pedigree ties, refinement |
| L9 | Deterministic replacement of a scheduling-dependent choice | The window-work witness is the first exhausting template in the count pass's declared order | refinement work account |
| L10 | Formula unification | Founder refit qualifies and re-checks two-row moves with one direct two-row gain | refit ascent |
| L11 | Formula unification | The selection context refit's inter-bin distance is computed through the map's interval distance | context refit |
| L12 | Formula unification | Pedigree emissions use `pow(x / max, η)`, skipping the power at η = 1, on every route | child emissions |
| L13 | Formula unification | Relatedness predicts and rescores a family's log-likelihood with one function of the member sum | relatedness family terms |
| L14 | Formula unification | The template linker is compiled without fast-math reassociation; it acts only when templates are rebuilt | template linker |
| L15 | Output representation | Unphased sample genotypes are written in ascending allele order, missing first, instead of 3.5's internal order | sample phase VCF |

Table: The numerical ledger for release 4.4.0: declared normalisations in which one arithmetic form replaces several, or a declared order replaces one chosen by workers, memory, object identity or the environment. No entry changes a model, objective, threshold, candidate set, search order or budget. Source: docs/developer/ties.md.
