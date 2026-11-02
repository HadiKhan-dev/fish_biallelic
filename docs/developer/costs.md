# Costs

Each stage's cost model: its work and storage terms, their size at K = 30, the cost exemptions, and what synthetic kernel ladders have measured. The table follows the specifications' cost sections and the code; it is not a benchmark.

## Symbols

N samples; K founder haplotypes; L sites (M selected sites); B bins or blocks; P sample pairs, N(N − 1)/2; E pedigree edges; F relatedness families; S = K + 1 painting and pedigree states (S′ ≤ 2K + 1 in the painting chain); C contigs, or class counts inside one stage. GB is decimal.

## What the columns establish

- **Work** and **storage** are algebraic models. The **K = 30** column evaluates their K-dependent factors or byte terms at K = 30, sometimes with stated N, L or B; it is a work or byte model, not a measured time.
- **Kernel slopes** come from ladders on direct synthetic inputs: K from 5 to 30 at N = 200, and N from 250 to 2,500 at K = 15, one thread, timed after a discarded first call. The slope is the fitted log–log exponent with its standard error. A K slope at or below 3.3 is accepted as at most cubic. A slope covers the probed kernel only; it does not bound the worst case of an adaptive search. A size proxy measures infrastructure, not a scientific K dependence.
- **K gate** is the verdict per stage: at most cubic when no exemption applies and every probed kernel's K slope is at most 3.3; exempt when a decided exemption (0, 0b, 0c, 0d) covers the stage, which is then measured and costed; exception 0e for the four documented above-cubic worst cases, whose bound is analytic; no intrinsic K for infrastructure measured on a size proxy. A provisional gate is rerun when the owning package lands.
- **Whole-run time and memory** of the release candidate are measured against 3.5 on the same inputs and node at the candidate freeze. Those results are not yet in this table; nothing here claims them.

## Exemptions and exceptions

The [accretion register](accretions.md#cost-exemptions) states decisions 0, 0b, 0c and 0d. Four further worst-case costs above cubic in K are carried exactly as documented exceptions under owner decision 0e, the treatment of 0c, and listed for review in release 4.1:

| Mechanism | Worst case |
|---|---|
| [AJ17](../methods/assembly.md#aj17) assembly joins' structural screening | a quartic trend in K in the direct query count on a finite grid |
| [RS05](../methods/refinement.md#rs05), [RS07](../methods/refinement.md#rs07) refinement window fixing | bounded only by the work account |
| [PR17](../methods/painting.md#pr17) painting label-continuity checks | O(K⁴) checks |
| [FR04](../methods/refit.md#fr04) founder refit ties | O(K⁴) flips and O(K⁵) swaps per phase; the direct two-row gain (L10) gives O(K⁴) over K rounds |

None of these binds on the validation inputs. Bounded exponential candidate families, such as shadow products under their combination budgets, are bounded by named parameters in the [parameter reference](../reference/parameters.md).

## Stage table

| Stage | Work | Storage (bytes) | K class | At K = 30 | K gate |
|---|---|---|---|---|---|
| [evidence](../methods/evidence.md) | O(N L); calibration O(N C) times fixed budgets | 33 N L retained; 16 N L permutation | no K term | K-independent | ≤ cubic: largest probed K slope -0.00 |
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
| [assembly.shadow_refits](../methods/assembly.md) | ≤2048 O(N m K²) shadow scores; 9 ranked refits | 8 N m(K+2)+2^26+8 N×8192/refit | exemption 0b; bounded shadow family | 9 refits; 2048×900=1843200 N m; ≈1.3GB/refit N2500 | exempt (0b); measured and costed |
| [assembly.orchestration_release](../methods/assembly.md) | O(N proxy_sites) partition; O(K B) composition; O(B A K log K) release | 256MiB+24 N L+component emissions+8 N bins(K+2)+moves/beams | ≤ K³ | 16 branch cap; the other assembly rows are separate | ≤ cubic: largest probed K slope 0.10 |
| [refinement.search](../methods/refinement.md) | O(N L K²) messages; O(B N K³) exchanges; canonical fixing O(K W R×programme) | acceptance≈2.5GB; polish 8 N Q(6h+12+b)+8 N b²/thread | above cubic: exception 0e ([RS05](../methods/refinement.md#rs05), [RS07](../methods/refinement.md#rs07)) | K30 N2500 polish≈3.7GB; canonical-fixing worst case carried; 4.1 candidate | exception 0e: analytic worst case, not bounded by the probed kernel slope; provisional until the package lands |
| [refinement.count](../methods/refinement.md) | 9(d+1) fits,up toK(d+1) tied; K shadow decisions; ≤16 histories | 24(B+1)N Q_(K+1)+32MiB/copy; streamed 512MiB | exemption 0b; bounded shadow family | K30 N2500 B641 copy≈19GB; ≤2048 shadow scores/member | exempt (0b); measured and costed; provisional until the package lands |
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
| [engine_graph](../methods/execution.md) | O(tasks)/boundary; O(contigs nodes) graph | O(tasks+resident outputs) | no intrinsic K/N | task-count proxy; overhead separate | no intrinsic K (operational size proxy) |
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
| assembly.shadow_refits | `assembly.count_rule` | K count decisions with default bootstrap; exemption 0b ranked refits separate | 1.00 ± 0.01 over 5–30 (exempt) | 1.28 ± 0.08 over 250–2,500 (exempt) |
| assembly.orchestration_release | `assembly.release` | full release of four-block K-row component without alternatives; no intrinsic N | 0.10 ± 0.01 over 5–30 (≤ cubic) | -0.01 ± 0.04 over 250–2,500 (measured) |
| refinement.search | `refinement.score` | direct exact whole-site score L64; canonical-fixing complexity remains unresolved | 1.71 ± 0.06 over 5–30 (≤ cubic) | 1.00 ± 0.00 over 250–2,500 (measured) |
| refinement.count | `refinement.count_bound` | exact one-deletion upper bounds in four blocks with K+1 rows; exemption 0b refits separate | 1.85 ± 0.12 over 5–30 (exempt) | 1.00 ± 0.00 over 250–2,500 (exempt) |
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
