# Costs

Each stage's cost model: its work and storage terms, their size at K = 30, the cost exemptions, and what synthetic kernel ladders have measured. The table follows the specifications' cost sections and the code; it is not a benchmark.

## Symbols

N samples; K founder haplotypes; L sites (M selected sites); B bins or blocks; P sample pairs, N(N − 1)/2; E pedigree edges; F relatedness families; S = K + 1 painting and pedigree states (S′ ≤ 2K + 1 in the painting chain); C contigs, or class counts inside one stage. GB is decimal.

## What the columns establish

- **Work** and **storage** are algebraic models. The **K = 30** column evaluates their K-dependent factors or byte terms at K = 30, sometimes with stated N, L or B; it is a work or byte model, not a measured time.
- **Kernel slopes** come from ladders on direct synthetic inputs: K from 5 to 30 at N = 200, and N from 250 to 2,500 at K = 15, one thread, timed after a discarded first call. The slope is the fitted log–log exponent with its standard error. A K slope at or below 3.3 is accepted as at most cubic. A slope covers the probed kernel only; it does not bound the worst case of an adaptive search. A size proxy measures infrastructure, not a scientific K dependence.
- **K gate** is the verdict per stage: at most cubic when no exemption applies and every probed kernel's K slope is at most 3.3; exempt when a decided exemption (0, 0c, 0d) covers the stage, which is then measured and costed; exception 0e for the four documented above-cubic worst cases, whose bound is analytic; no intrinsic K for infrastructure measured on a size proxy. A provisional gate is rerun when the owning package lands.
- **Whole-run time and memory** of the release candidate are measured against 3.5 on the same inputs and node at the candidate freeze. Those results are not yet in this table; nothing here claims them.

## Exemptions and exceptions

The [accretion register](accretions.md#cost-exemptions) states decisions 0, 0c and 0d. Four further worst-case costs above cubic in K are carried exactly as documented exceptions under owner decision 0e, the treatment of 0c. Their 4.2 review is in the table:

| Mechanism | Worst case |
|---|---|
| [AJ17](../methods/assembly.md#aj17) assembly joins' structural screening | a quartic trend in K in the direct query count on a finite grid; the adaptive route the pipeline uses has a cubic analytic bound, including chunking |
| [RS05](../methods/refinement.md#rs05), [RS07](../methods/refinement.md#rs07) refinement window fixing | bounded only by the work account; no component stopped on it in the 55-case release matrix |
| [PR17](../methods/painting.md#pr17) painting label-continuity checks | O(J K²) candidates per iteration, each a full O(N(B+J)K²) decode; tie peers grouped once per iteration (O(Q)); repair iterations are bounded by the integer switch decrease |
| [FR04](../methods/refit.md#fr04) founder refit ties | O(K⁴) flips and O(K⁵) swaps per phase; the direct two-row gain (L10) gives O(K⁴) over K rounds |

The PR17 and FR04 decisions act on the validation inputs (93 released continuity moves, 14 refit ambiguity cells across the 4.0 matrix), with small classes; their above-cubic worst cases are analytic and are not certified by those workloads. An FR04 reorganisation with identical decisions (cached quadratic two-row gains, exact fallback at numerical boundaries) was prototyped for 4.2; it does not lower the worst-case bound and was not adopted. Bounded exponential candidate families, such as shadow products under their combination budgets, are bounded by named parameters in the [parameter reference](../reference/parameters.md).

## Natural-domain count refits

4.2 removes the nine-candidate admission cap (former exemption 0b). Refinement fits all eligible deletions and every duplicate addition seed; AJ23 fits a duplicate of every selected founder whenever its existing seed route is triggered. At K = 10, 20 and 30 the full domain can contain 10, 20 and 30 candidates, respectively. Private-cell eligibility and certified whole-round bounds retain their existing meaning.

Each round fits O(K) candidates, costing O(N L K³) under the quadratic fixed-K work model. A deletion pass has at most d + 1 rounds for d accepted deletions, and d = O(K): its worst-case count-refit work is O(N L K⁴). Successive additions and join count changes also multiply per-round work. These estimates factor out fixed-K search iterations, canonical window programmes, shadow families and continuation histories; they do not prove that those additional operations are quadratic or that a complete stage is cubic. There is no admission cap and no cost exemption for it.

The 4.0 saved-output census found 25 deletion rounds in 25 k30 passes (one each), 1,268 in 1,157 TroMau passes (1,066 with one, 77 with two, 11 with three and three with five), and 1,359 in 1,347 std passes (1,335 with one and 12 with two). These are recorded component passes, not independent biological replicates or a bound on future inputs.

The R1 replay evidence establishes a TroMau K10 deletion whose omitted founder 1 improves endpoint F by 54.932936 nats. The smaller-count verdict stays unchanged but the preferred deletion panel changes. A k30 addition also has an omitted endpoint improving F by 103.094865 nats; that addition is still rejected. These are fixed-round endpoint comparisons, not a new pipeline release. Replays of the 4.2 implementation must establish its complete endpoint selection and measure extra fits and time. RC14 cleanup fired zero times in 54,123 final-role component records; its removal avoids a separate deletion path and its repeated unrefitted scoring.

## Stage table

| Stage | Work | Storage (bytes) | K class | At K = 30 | K gate |
|---|---|---|---|---|---|
| [evidence](../methods/evidence.md) | GL O(N L); encoding O(N L log L); calibration O(N C) times fixed budgets | joint storage < 33 N L when coded; one-sample O(L + D) planning | no K term | K-independent | ≤ cubic: largest probed K slope -0.00 |
| [relatedness.families](../methods/relatedness.md) | O(N² M) screen; O(N² F) per sweep; O(F N M) support | 192 N M concatenate; 64 N M screens | no K term | F is families, not founders | ≤ cubic: largest probed K slope -0.01 |
| [relatedness.pairs](../methods/relatedness.md) | O(P L/10), P=N(N-1)/2; optional | 72 P | exemption 0d | N2500:3123750 pairs,224910000 bytes; off by default | exempt (0d); measured and costed |
| [discovery.fit](../methods/discovery.md) | O(N L K²) trajectory; O(N L K⁴) whole fitting term | 32 N×16384 transition cache; 24 B×1024 N patterns | exemption 0c | K⁴=810000; 50 iterations; ≤2 trajectories/start | exempt (0c); measured and costed |
| [discovery.search](../methods/discovery.md) | O(N L K⁴) fitting; O(N L K³) cavity; bounded exact cuts | 18 R L residual; 8 N S tile | exemption 0c | 810000 N L; exact switch 12; 48/96 replacements; 6 panels/K; 50 iterations | exempt (0c); measured and costed |
| [discovery.release](../methods/discovery.md) | O(N K L) release; O(K L log K) keys | 32 N×16384 plus patterns | ≤ K³ | linear release factor 30 | ≤ cubic: largest probed K slope 0.13 |
| [selection.fit](../methods/selection.md) | O(N L K²(K+L)) backstop; context snaps O((20+4K+2L)²) passes | 8 threads×64 L(S(S+1)/2+S); 1024 L S² release | exemption 0c | K²=900; snaps(140+2L)²; L400 release 393625600 bytes | exempt (0c); measured and costed |
| [selection.search](../methods/selection.md) | O(N L K³(K+L)) search; O(K⁵) context screening | 232 N L+workspace; context 8 N B S² | exemptions 0 and 0c | K⁵=24300000/round; 27000 N L(30+L); N2500 B100 S31=1922000000 bytes | exempt (0 and 0c); measured and costed |
| [assembly.completion](../methods/assembly.md) | O(N L K²) cross-fit; O(N K³) core sites; O(N K²) links | 25 N L+40 N K² min(L,50)+24 bins N Q+64 N Q+48MiB | ≤ K³ | N2500 K30 bin 50:900000000 bytes/array; 4500000000 with temporaries | ≤ cubic: largest probed K slope 1.62 |
| [assembly.site_model](../methods/assembly.md) | O(I N K²) BP I≤500; O(N K³) radius 3 ball; bounded 2^10 cube | 8 N K² quadratics+support | ≤ K³ | 4526 radius 3 configurations; 500×900×N BP work/site | ≤ cubic: largest probed K slope 3.06 |
| [assembly.linker](../methods/assembly.md) | O(N E K²) scans; O(N K³) edge propagation | 28 N E+8 N E C; 16 N K² per block/gap | ≤ K³ | 27000 cubic factor; ≤51 evaluations/gap | ≤ cubic: largest probed K slope 1.90 |
| [assembly.joins](../methods/assembly.md) | O(N m² K³) structural per query; direct query-count finite-grid quartic trend | 8 N sum(P_j m_j) emissions; 8 N m K exclusions | above cubic: exception 0e ([AJ17](../methods/assembly.md#aj17)) | K30 N2500 m2000 emissions≈18.6GB; direct-route finite-grid quartic trend carried; 4.1 candidate | exception 0e: analytic worst case, not bounded by the probed kernel slope |
| [assembly.shadow_refits](../methods/assembly.md) | ≤2048 O(N L K²) shadow scores; O(K) seed fits, O(N L K³)/round | 8 N m(K+2)+2^26+8 N×8192/refit | natural domain; repeated rounds can be quartic | 30 seed fits/round; shadow budgets unchanged | 4.2 accepted count cost; kernel slope does not bound a whole pass |
| [assembly.orchestration_release](../methods/assembly.md) | O(N proxy_sites) partition; O(K B) composition; O(B A K log K) release | 256MiB+24 N L+component emissions+8 N bins(K+2)+moves/beams | ≤ K³ | 16 branch cap; the other assembly rows are separate | ≤ cubic: largest probed K slope 0.10 |
| [refinement.search](../methods/refinement.md) | O(N L K²) messages; O(B N K³) exchanges; canonical fixing O(K W R×programme) | acceptance≈2.5GB; polish 8 N Q(6h+12+b)+8 N b²/thread | above cubic: exception 0e ([RS05](../methods/refinement.md#rs05), [RS07](../methods/refinement.md#rs07)) | K30 N2500 polish≈3.7GB; canonical-fixing worst case carried; 4.1 candidate | exception 0e: analytic worst case, not bounded by the probed kernel slope; provisional until the package lands |
| [refinement.count](../methods/refinement.md) | O(K) fits/round, O(N L K³); ≤d+1 deletion rounds, O(N L K⁴) worst case; ≤16 histories | candidate fits plus existing message and shadow workspaces; admission-only copy messages removed | natural domain; quartic worst-case pass | up to 30 eligible fits/round; ≤2048 shadow scores/member | 4.2 accepted count cost; no whole-pass cubic claim |
| [painting.coalescence](../methods/painting.md) | O(N B S²) messages; O(links N(S² log S+C² S)); worst K⁴ checks | 0.6–0.8GB/sample at K30 B10000 | above cubic: exception 0e ([PR17](../methods/painting.md#pr17)) | N2500 K30 links 2000≈1e11 gain operations; quartic candidate checks carried; 4.1 candidate | exception 0e: analytic worst case, not bounded by the probed kernel slope |
| [painting](../methods/painting.md) | O(N L S²) emissions; O(N B S'²) chain; O(N B S³) masks; S'≤2K+1 | 40 S'² B+16 S² B+24 L/sample; pointers 4 S'² B | ≤ K³ | S'=61; pointers 148840000 bytes B10000; ≈1.6GB/sample | ≤ cubic: largest probed K slope 1.94 |
| [founder_refit](../methods/refit.md) | O(N+K³ log K)/site/sweep; direct two-row worst K⁴; ties K⁴ flips/K⁵ pairs | 4 N L label expansion | above cubic: exception 0e ([FR04](../methods/refit.md#fr04)) | K⁴=810000; K⁵=24300000/site/phase; observed≤3 rounds,≤2 tie members; 4.1 candidate | exception 0e: analytic worst case, not bounded by the probed kernel slope |
| [pedigree.prepare](../methods/pedigree.md) | O(N B S²) sources; O(N M S) ALT; O(N B(S²+128S)) bridges | 24 N B S²+1.25 evidence+painting+2GiB | ≤ K³ | S31:right weights 7688 bytes/sample/bin | ≤ cubic: largest probed K slope 1.79 |
| [pedigree.screen_pair_bound](../methods/pedigree.md) | O(R² B G³) pair bound G≤S+1; O(C² P) panel; O(D L) exclusion | 16 N(B-1)S² geometry; 4 P/contig; stack streamed | ≤ K³ | N2500 B1000 S31≈38GB geometry; 414M pairs≈1.7GB/contig | ≤ cubic: largest probed K slope 2.17 |
| [pedigree.scores_exposure](../methods/pedigree.md) | O((E/2+T)B S(S+1)/2) exposure; O(N B J²) junctions; O(C N²) order/fit | 8 N B S(S+1)/2 masses; 32 N² outputs; 40 N² family | ≤ K³ | S31:496 features×3123750 unordered edges×B | ≤ cubic: largest probed K slope 1.58 |
| [pedigree.decision](../methods/pedigree.md) | O(C(N²+T)) stack; O(terms)/calibration; graph O(N(N+edges)+A) | four 8 C E arrays+likelihood/junction; N² dominant | no K term | N2500 C22 each 8 C E≈1.1GB | ≤ cubic: largest probed K slope 0.00 |
| [pedigree.resampling_release](../methods/pedigree.md) | ≤1000+C fits; O(C A+C N²+I(A+N²)+R A+R N²+R² T+graph)/fit | ≈80 N²/worker | no K term | N2500≈500000000 bytes/fit worker | ≤ cubic: largest probed K slope 0.01 |
| [phase.solve](../methods/phase.md) | O(B(N S³+S² sum A_n))/sweep; O(children B S³) pairing | 8 N S² B emissions; O(B(S+2)²G) pointers | ≤ K³ | N2500 S31 B3000–3300:57.66–63.426GB emissions | ≤ cubic: largest probed K slope 1.93 |
| [phase.release](../methods/phase.md) | O(N B(S+2)³ G) posterior; O(N L S²) site; O(E B) chains | 12 N L spool+5 N L site+E | ≤ K³ | K30 S31 L330000≈3e8 pair visits/sample | ≤ cubic: largest probed K slope 2.81 |
| [recombination](../methods/recombination.md) | O(E M) following; O(E Bm+calls) aggregate; O(loci² E+loci² parents grid) | 16 E Bm+phase chains | no K term | quadratic loci detector separate | ≤ cubic: largest probed K slope -0.03 |
| [sites_export](../methods/export.md) | O(M) scan; O(N M) sample VCF; O(M(C+K)) founder VCF; O(B K²) flags | 2 N L calls; O(M) sites | ≤ K² | N2500 M400000:1e9 cells; calls 2GB | ≤ cubic: largest probed K slope 0.01 |
| [reports](../methods/export.md) | O(N chunks K) painting writer; O(N²) optional pairs | one decoded product+buffers | ≤ K; optional pairs: exemption 0d | N2500 optional 3123750 rows | ≤ cubic: largest probed K slope 0.90 |
| [simulation](../methods/simulation.md) | O(N L) transmissions/reads; linear founders | 2 N L/contig haplotypes; 48 N L reads | ≤ K | N2500 L400000 C22 all-resident 44GB | ≤ cubic: largest probed K slope 0.07 |
| [evaluation](../methods/evaluation.md) | O(sum K L F) counts; O(K³) assignment; O(N L) phase; O(windows F²×100) | calls/truth/mask; no reordered calls copy | ≤ K³ | K30 F≤60; assignment 27000 units | ≤ cubic: largest probed K slope 1.24 |
| [contracts](../methods/core.md) | O(K L) views; O(B K²) partner changes | O(K L) panels; N L founder view | ≤ K² | partner pair factor 900 | ≤ cubic: largest probed K slope 0.86 |
| [core.ties_numerics](../methods/core.md) | O(n log n) sort; O(C n) sum; O(n) expansion | O(n) scratch | no intrinsic K/N | array-size proxy only | no intrinsic K (operational size proxy) |
| [kernel_warmup](../methods/execution.md) | O(reachable signatures) compilation; worker parallel | compiler/cache dependent | no intrinsic K/N | compilation of reachable signatures; no intrinsic K | no intrinsic K (operational size proxy) |
| [store](../methods/execution.md) | O(bytes) encode/decode; selective items | bounded in-flight payload | no intrinsic K/N | payload-size proxy only | no intrinsic K (operational size proxy) |
| [engine_graph](../methods/execution.md) | O(tasks)/boundary; O(contigs nodes) graph | O(tasks + live expansion results + public outputs); geometric ledger growth | no intrinsic K/N | task-count proxy; overhead separate | no intrinsic K (operational size proxy) |
| [interface](../methods/execution.md) | Parsing/validation; negligible | O(configuration) | no intrinsic K/N | settings-resolution proxy only (interface.settings); no scientific K ladder | no intrinsic K (operational size proxy) |

## Kernel ladders

One row per probed kernel. Stages without a row have no ladder result yet.

| Stage | Probe | What it covers | K slope | N slope |
|---|---|---|---|---|
| evidence | `evidence.calibration_objective` | sample objective/gradient at 230 depth-alt cells; no intrinsic K | -0.00 ± 0.00 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| evidence | `evidence.likelihoods` | read-model likelihood application N×256; no intrinsic K | -0.00 ± 0.00 over 5–30 (≤ cubic) | 0.98 ± 0.02 over 250–2,500 (measured) |
| evidence | `evidence.quality_genotypes` | genotype expectations, default 60 iterations, N×256; no intrinsic K | -0.00 ± 0.00 over 5–30 (≤ cubic) | 1.12 ± 0.01 over 250–2,500 (measured) |
| relatedness.families | `relatedness.screen` | all singleton family screen; F=N; no intrinsic K | -0.02 ± 0.01 over 5–30 (≤ cubic) | 2.00 ± 0.00 over 250–2,500 (measured) |
| relatedness.families | `relatedness.terms` | AF and family terms; no intrinsic K | -0.01 ± 0.01 over 5–30 (≤ cubic) | 1.02 ± 0.04 over 250–2,500 (measured) |
| relatedness.pairs | `relatedness.pairs` | all pairs, 16 thinned loci; no intrinsic K; exemption 0d | 0.03 ± 0.01 over 5–30 (exempt) | 1.99 ± 0.00 over 250–2,500 (exempt) |
| discovery.fit | `discovery.assignment` | full fixed-K A step L32, cached evidence patterns; K² per assignment | 1.57 ± 0.08 over 5–30 (≤ cubic) | 0.95 ± 0.03 over 250–2,500 (measured) |
| discovery.fit | `discovery.fit_family` | complete two-schedule fit family L32; fresh fit memos; default 50 iterations; exemption 0c | 1.28 ± 0.08 over 5–30 (exempt) | 0.91 ± 0.05 over 250–2,500 (exempt) |
| discovery.search | `discovery.cavity` | held-out cavity L32/default 100 iterations; search budget exemption 0c remains analytic | 2.58 ± 0.30 over 5–30 (exempt) | 0.15 ± 0.12 over 250–2,500 (exempt) |
| discovery.release | `discovery.release` | fixed-assignment founder release K×32 with N carriers | 0.13 ± 0.01 over 5–30 (≤ cubic) | 1.03 ± 0.03 over 250–2,500 (measured) |
| selection.fit | `selection.exact_pass` | EXACT forward/backward pass L32, memo-free; per-pass K2 | 1.62 ± 0.06 over 5–30 (≤ cubic) | 0.99 ± 0.01 over 250–2,500 (measured) |
| selection.fit | `selection.fit_fixed` | full fit L32 with unchanged default backstop; exemption 0c; adaptive path | 1.64 ± 0.10 over 5–30 (exempt) | 0.96 ± 0.01 over 250–2,500 (exempt) |
| selection.search | `selection.context_matching` | K² context matching kernels of K×K costs per round; exemption 0; no intrinsic N | 4.01 ± 0.05 over 5–30 (exempt) | 0.01 ± 0.01 over 250–2,500 (exempt) |
| assembly.completion | `assembly.cavity_fill` | whole-bin cavity fill L64, ten-percent unknown calls, default budgets | 1.62 ± 0.13 over 5–30 (≤ cubic) | 0.96 ± 0.02 over 250–2,500 (measured) |
| assembly.completion | `assembly.crossfit` | two-fold crossfit L32, ten-percent unknown calls, default budgets | 1.34 ± 0.08 over 5–30 (≤ cubic) | 0.81 ± 0.03 over 250–2,500 (measured) |
| assembly.site_model | `assembly.site_posterior` | weak informative site, all K uncertain; exact/core route recorded | 3.06 ± 0.11 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| assembly.linker | `assembly.linker_scan` | forward micro-HMM L32 with nonuniform priors; no zero-prior memo reuse | 1.90 ± 0.03 over 5–30 (≤ cubic) | 0.98 ± 0.02 over 250–2,500 (measured) |
| assembly.joins | `assembly.join_score` | exact panel score, four blocks of four bins; structural-query count remains unresolved | 1.75 ± 0.06 over 5–30 (≤ cubic) | 1.02 ± 0.02 over 250–2,500 (measured) |
| assembly.shadow_refits | `assembly.count_rule` | K count decisions with default bootstrap; natural-domain fits and repeated rounds separate | 1.00 ± 0.01 over 5–30 (kernel only) | 1.28 ± 0.08 over 250–2,500 (kernel only) |
| assembly.orchestration_release | `assembly.release` | full release of four-block K-row component without alternatives; no intrinsic N | 0.10 ± 0.01 over 5–30 (≤ cubic) | -0.01 ± 0.04 over 250–2,500 (measured) |
| refinement.search | `refinement.score` | direct exact whole-site score L64; canonical-fixing complexity remains unresolved | 1.71 ± 0.06 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| refinement.count | `refinement.count_bound` | exact one-deletion upper bounds in four blocks with K+1 rows; natural-domain fits and repeated rounds separate | 1.85 ± 0.12 over 5–30 (kernel only) | 1.00 ± 0.00 over 250–2,500 (kernel only) |
| painting.coalescence | `painting.transposition_gains` | all K(K-1)/2 transposition gains at two links; worst candidate rechecks remain analytic | 2.39 ± 0.05 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| painting | `painting.chain` | structured sum-product, N×8 bins, Sprime=2K+1, masses enabled | 1.94 ± 0.01 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| painting | `painting.decode_masks` | both Viterbi conventions plus interval/canonical masks, N×8 bins | 1.73 ± 0.02 over 5–30 (≤ cubic) | 1.02 ± 0.00 over 250–2,500 (measured) |
| founder_refit | `refit.alleles` | conditional refit N×32, default three sweeps; random mismatches, not adversarial tie classes | 0.88 ± 0.08 over 5–30 (≤ cubic) | 0.71 ± 0.03 over 250–2,500 (measured) |
| pedigree.prepare | `pedigree.sources` | source factors and marginals, N×8 bins×S² | 1.79 ± 0.06 over 5–30 (≤ cubic) | 1.07 ± 0.03 over 250–2,500 (measured) |
| pedigree.screen_pair_bound | `pedigree.pair_bound` | all pairs of N-1 candidate parents for one child, two bins, G=S | 2.17 ± 0.16 over 5–30 (≤ cubic) | 2.00 ± 0.00 over 250–2,500 (measured) |
| pedigree.scores_exposure | `pedigree.exposure` | all N(N-1)/2 edges plus N trios, two bins, all S(S+1)/2 features | 1.58 ± 0.05 over 5–30 (≤ cubic) | 2.00 ± 0.00 over 250–2,500 (measured) |
| pedigree.decision | `pedigree.family_messages` | dense N² M0/M1 alternative family log-odds pass; iterative fit separate; no intrinsic K | 0.00 ± 0.01 over 5–30 (≤ cubic) | 2.01 ± 0.02 over 250–2,500 (measured) |
| pedigree.resampling_release | `pedigree.resample_counts` | full configured bootstrap replicate accumulator, fit kernel separate; no intrinsic K | 0.01 ± 0.01 over 5–30 (≤ cubic) | 1.71 ± 0.31 over 250–2,500 (measured) |
| phase.solve | `phase.viterbi` | all N root-sample blocks, eight bins, S=K+1; descendant paths separate | 1.93 ± 0.04 over 5–30 (≤ cubic) | 1.01 ± 0.00 over 250–2,500 (measured) |
| phase.release | `phase.posterior` | all N conditional sum-product blocks, eight bins, two gauges | 2.81 ± 0.02 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| recombination | `recombination.follow` | N meiosis chain following, 256 bins; no intrinsic K | -0.03 ± 0.02 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| sites_export | `outputs.samples` | actual BCF sample writer N×256; no intrinsic K | 0.01 ± 0.01 over 5–30 (≤ cubic) | 0.90 ± 0.03 over 250–2,500 (measured) |
| reports | `outputs.painting_report` | actual founder support tables; all K pooled rows; N×K×256 | 0.90 ± 0.02 over 5–30 (≤ cubic) | 0.97 ± 0.00 over 250–2,500 (measured) |
| simulation | `simulation.reads` | independent-site read generator N×256; K founder source only | 0.07 ± 0.01 over 5–30 (≤ cubic) | 0.99 ± 0.01 over 250–2,500 (measured) |
| evaluation | `evaluation.founders` | full founder evaluator K rows,F=2K,L=256; no intrinsic N | 1.24 ± 0.11 over 5–30 (≤ cubic) | -0.00 ± 0.01 over 250–2,500 (measured) |
| contracts | `contracts.partners` | partner changes B=32; no intrinsic N | 0.86 ± 0.07 over 5–30 (≤ cubic) | 0.01 ± 0.01 over 250–2,500 (measured) |
| core.ties_numerics | `core.ties` | workload proxy length=N*K; not a scientific K gate | 1.07 ± 0.03 over 5–30 (size proxy) | 1.10 ± 0.00 over 250–2,500 (size proxy) |
| kernel_warmup | `core.kernel_manifest` | signature encode/decode proxy count=N*K; cold compilation separate | 1.00 ± 0.00 over 5–30 (size proxy) | 1.00 ± 0.00 over 250–2,500 (size proxy) |
| store | `store.codec` | payload encode/decode proxy [N,K,32]; no scientific K gate | 0.90 ± 0.03 over 5–30 (size proxy) | 1.03 ± 0.02 over 250–2,500 (size proxy) |
| engine_graph | `engine.graph` | independent-task validation proxy count=N*K; not scheduler throughput | 1.98 ± 0.01 over 5–30 (size proxy) | 1.99 ± 0.00 over 250–2,500 (size proxy) |
| interface | `interface.settings` | settings resolution proxy: TOML load, merge, typed build, validation and settings_text of a simulation configuration with cohorts (K, N) negligible; no intrinsic N or K | 0.00 ± 0.00 over 5–30 (size proxy) | -0.00 ± 0.01 over 250–2,500 (size proxy) |

All stages have a ladder.


Execution memory and CPU boundaries (4.4): `ResourceRequest.cost` orders ready
work; measured wall cost per stage/callable is its fallback. Neither changes a
scientific work allowance. Admission uses measured working set and physical
availability, with gradually spaced starts only above 90% use. The ledger's
initial 4,096 token/request and 256 label slots double independently; growth copies
one table and preserves slot indices across processes. Old appended table space
is less than twice the latest capacity per table and is removed with the run ledger.

Expansion children are private intermediate results by default. Their completed
values are released after the gather's last declared read, while completion keys
remain for dependencies. Caller-supplied task results remain returned;
`retain_intermediates=True` is the diagnostic retention route. A later dynamic
consumer of an already-released private result must request that route explicitly.
Resident input caches loaded by `resident_or_load` can be evicted while idle;
explicit `retain` state and kept gathers cannot. Clearing a cache reference does
not invalidate the same value held by a live continuation. Idle pressure cleanup
trims and, when no pinned/shared owned state remains, replaces the worker once.

Use `engine.scheduler.kernel_team(max_threads=width)` at each parallel kernel or
independent section. A task can grow beyond its initial team as peers finish, but
cannot revoke a peer's running kernel claim. The sum of protected process claims
stays within the CPU authority. The parallel width, ordered placement, sample
traversal, reductions and random streams must preserve the scientific computation.


Shared-value transport also reuses backing across independent bundles. Immutable
private views with the same address and byte extent share one copied segment;
weak source views prevent address reuse from selecting stale bytes without keeping
the private twin alive. Read-only mapped/shared arrays pin the exact device/inode
through a read-only descriptor and carry byte offset, shape and dtype separately.
A bundle reference count keeps the backing until its last consumer. Existing file
and shared mappings incur no new data segment. Linux proc-fd paths preserve access
after the source mapping closes or its pathname disappears. Writable file maps and
private copy-on-write mappings are copied because their current bytes may differ
from the underlying file. The engine's mapped storage route remaps read-only after
filling, so it can participate in this transport without another copy.

## Residual discovery clique reuse (4.4)

Residual alternative groups retain the content-ordered maximal greedy extension of
**every compatible edge**. The implementation memoizes extensions by the exact
closed common neighbourhood of the two seed vertices. Both endpoints are universal
within that neighbourhood: inserting them at the start or at their content-order
positions cannot exclude another candidate. Consequently, the greedy extension is
a function of the neighbourhood alone. Equal bitsets reuse the extension; hash
collisions compare all words. This changes neither the groups offered for size-class
screening nor any omission or decision. The intersection receipt still counts the
original work for every seed edge, including reused extensions.

For A vertices, E edges and W = ceil(A/64) packed words, key construction costs
O(E W). Only distinct closed common neighbourhoods run the O(A W) extension.
Worst-case work and storage remain polynomial with the existing bounds; the memo
adds O(E W + E) temporary storage. Dense graphs with repeated neighbourhoods avoid
rebuilding and sorting thousands of identical large cliques. Sparse graphs retain
the same domain. This is execution reuse within DI18, not a change to exemption
0c's replacement, beam, convergence or stopping contracts.

## Discovery timing comparisons

The block diagnostic timer surrounds `discover_block`. Its CPU and wall values
exclude the contig reader, task transport, result encoding and checkpoint
publication. Whole-task timing includes additional work; the 4.3 driver reads
one evidence product per contig before expanding it into block tasks.

CPU seconds count scheduled processor time, not a frequency-normalised amount of
work. CPU/wall near one therefore establishes neither equal clock rates nor equal
hardware throughput. Compare matched blocks on the same host, inspect effective
clock rates, separate first-use kernel loading/compilation from warm execution,
and check one versus many workers before attributing a cross-host depth ratio to
the scientific search. Keep all of those measurements separate from the search's
candidate, iteration and logical-work receipts.

## Evidence dictionary windows (4.4)

Encoding sorts one sample's L pair keys and repeats bounded searchsorted passes
against D distinct keys: O(N L log L + sum_n L log D_n), independent of K.
The actual strict joint-byte admission rule includes codes, pair/GL tables,
metadata and all dense fallback observation fields. It never compares GL bytes alone.

Resident GL/mask tiles target 8 MiB of projection and lookup workspace, at most
2048 sites; one site is always processed for arbitrarily large sample axes.
Checkpoint windows select/deduplicate only the codes they need and decode
consecutive dictionary-row runs. A W-site sample window costs O(W log W) lookup
work, O(W) scratch and at most min(D, W) decoded rows per dictionary field.
An interleaved N-sample pass therefore requests O(N L) dictionary bytes rather
than O(N D ceil(L/W)); physical I/O also depends on compression chunks and the
store's bounded shared decoded-chunk cache. A cache miss fully decompresses its
physical chunk before NumPy selects the requested bytes; library partial-range
decoding is not used. The cache holds at most 1 GiB of decoded data, evicting
before allocation (one oversized chunk may stand alone), plus the transient
compressed input of one miss. Scattered requests spanning more chunks than fit
can repeat whole-chunk decompression; requested dictionary-row bytes therefore
do not bound decompression work. No scientific work is omitted.

Dense painting fills its already-required output flags before reading GLs, then
walks sample tiles whose full rows share a physical GL chunk. A row crossing a
chunk boundary stands alone and walks forward through site windows. Tiles never
exceed the existing sample/team bound; whole-bin GL/mask scratch retains its
64 MiB target. This avoids alternating large GL and mask chunks or revisiting
many GL chunks on every small site window. Per-sample/bin accumulation and the
emission formula are unchanged. The decoded cache may still increase peak RSS
relative to caching compressed chunks; it is charged to the existing byte bound.
