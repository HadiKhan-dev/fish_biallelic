# Costs

Work and storage below describe the current algorithms. They are algebraic size models, not measured times or bounds on all adaptive search trajectories.

## Symbols

N samples; K founder haplotypes; L sites (M selected sites); B bins or blocks; P sample pairs, N(N − 1)/2; E pedigree edges; F relatedness families; S = K + 1 painting and pedigree states (S′ ≤ 2K + 1 in the painting chain); C contigs, or class counts inside one stage. GB is decimal.

## Exemptions and exceptions

The [mechanism register](accretions.md#cost-exemptions) identifies costly bounded searches and optional reports. Assembly structural screening, the local-selection context screen, refinement window fixing, the refinement count pass and founder-refit tie checks can exceed cubic work in founder count; painting label continuity is bounded per iteration by its check budget. Their method sections state the relevant candidate, frontier and work limits.

## Natural-domain count refits

Refinement fits all eligible deletions. Join duplicate-seed work fits a duplicate of every selected founder whenever its seed route is triggered. Private-cell eligibility and certified whole-round bounds define the admitted domain.

Each round fits O(K) candidates, costing O(N L K³) under the quadratic fixed-K work model. A deletion pass has at most d + 1 rounds for d accepted deletions, where d = O(K), giving O(N L K⁴) worst-case count-refit work. These terms exclude fixed-K search iterations, canonical window programmes, shadow families and continuation histories. There is no candidate admission cap or whole-pass cubic guarantee.

## Stage table

| Stage | Work | Storage (bytes) | K class | At K = 30 |
| --- | --- | --- | --- | --- |
| [evidence](../methods/evidence.md) | GL O(N L); encoding O(N L log L); calibration O(N C) times fixed budgets | joint storage < 33 N L when coded; one-sample O(L + D) planning | no K term | K-independent |
| [relatedness.families](../methods/relatedness.md) | O(N² M) screen; O(N² F) per sweep; O(F N M) support | 192 N M concatenate; 64 N M screens | no K term | F is families, not founders |
| [relatedness.pairs](../methods/relatedness.md) | O(P L/10), P=N(N-1)/2; optional | 72 P | exemption 0d | N2500:3123750 pairs,224910000 bytes; off by default |
| [discovery.fit](../methods/discovery.md) | O(N L K²) trajectory; O(N L K⁴) whole fitting term | 32 N×16384 transition cache; 24 B×1024 N patterns | exemption 0c | K⁴=810000; 50 iterations; ≤2 trajectories/start |
| [discovery.search](../methods/discovery.md) | O(N L K⁴) fitting; O(N L K³) cavity; O(K³ + K² W + T K² √K) gauge cuts (W≤N, T≤2000) | 18 R L residual; 8 N S tile | exemption 0c | 810000 N L; 16 GW starts at every K; 48/96 replacements; 6 panels/K; 50 iterations |
| [discovery.release](../methods/discovery.md) | O(N K L) release; O(K L log K) keys | 32 N×16384 plus patterns | ≤ K³ | linear release factor 30 |
| [selection.fit](../methods/selection.md) | O(N L K²(K+L)) backstop; context snaps O((20+4K+2L)²) passes | 8 threads×64 L(S(S+1)/2+S); 1024 L S² release | exemption 0c | K²=900; snaps(140+2L)²; L400 release 393625600 bytes |
| [selection.search](../methods/selection.md) | O(N L K³(K+L)) search; O(K⁴) context screening plus ≤128 O(K³) exact codes/round | 232 N L+workspace; context 8 N B S² | exemptions 0 and 0c | K⁴=810000/round; ≤128 exact codes; 27000 N L(30+L); N2500 B100 S31=1922000000 bytes |
| [assembly.completion](../methods/assembly.md) | O(N L K²) cross-fit; O(N K³) core sites; O(N K²) links | 25 N L+40 N K² min(L,50)+24 bins N Q+64 N Q+48MiB | ≤ K³ | N2500 K30 bin 50:900000000 bytes/array; 4500000000 with temporaries |
| [assembly.site_model](../methods/assembly.md) | O(I N K²) BP I≤500; O(N K³) radius 3 ball; bounded 2^10 cube | 8 N K² quadratics+support | ≤ K³ | 4526 radius 3 configurations; 500×900×N BP work/site |
| [assembly.linker](../methods/assembly.md) | O(N E K²) scans; O(N K³) edge propagation | 28 N E+8 N E C; 16 N K² per block/gap | ≤ K³ | 27000 cubic factor; ≤51 evaluations/gap |
| [assembly.joins](../methods/assembly.md) | O(N m² K³) structural per query; direct query-count finite-grid quartic trend | 8 N sum(P_j m_j) emissions; 8 N m K exclusions | above cubic: exception 0e ([AJ17](../methods/assembly.md#aj17)) | K30 N2500 m2000 emissions≈18.6GB; direct-route finite-grid quartic trend carried |
| [assembly.shadow_refits](../methods/assembly.md) | ≤2048 O(N L K²) shadow scores; O(K) seed fits, O(N L K³)/round | 8 N m(K+2)+2^26+8 N×8192/refit | natural domain; repeated rounds can be quartic | 30 seed fits/round; shadow budgets unchanged |
| [assembly.orchestration_release](../methods/assembly.md) | O(N proxy_sites) partition; O(K B) composition; O(B A K log K) release | 256MiB+24 N L+component emissions+8 N bins(K+2)+moves/beams | ≤ K³ | 16 branch cap; the other assembly rows are separate |
| [refinement.search](../methods/refinement.md) | O(N L K²) messages; O(B N K³) exchanges; canonical fixing O(K W R×programme) | acceptance≈2.5GB; polish 8 N Q(6h+12+b)+8 N b²/thread | above cubic: exception 0e ([RS05](../methods/refinement.md#rs05), [RS07](../methods/refinement.md#rs07)) | K30 N2500 polish≈3.7GB; canonical-fixing worst case carried |
| [refinement.count](../methods/refinement.md) | O(K) fits/round, O(N L K³); ≤d+1 deletion rounds, O(N L K⁴) worst case; ≤16 histories | candidate fits plus existing message and shadow workspaces; one 64MiB score-miss ledger per component; admission-only copy messages removed | natural domain; quartic worst-case pass | up to 30 eligible fits/round; ≤2048 shadow scores/member |
| [painting.coalescence](../methods/painting.md) | O(N B S²) messages; O(links N(S² log S+C² S)); per iteration ≤b charged checks and ≤2b−1 trial decodes, O(b N (B+J) K²); iterations ≤ the initial switch total | 0.6–0.8GB/sample at K30 B10000 | ≤ K³ per iteration ([PR17](../methods/painting.md#pr17)) | N2500 K30 links 2000≈1e11 gain operations; b=64 |
| [painting](../methods/painting.md) | O(N L S²) emissions; O(N B S'²) chain; O(N B S³) masks; S'≤2K+1 | 40 S'² B+16 S² B+24 L/sample; pointers 4 S'² B | ≤ K³ | S'=61; pointers 148840000 bytes B10000; ≈1.6GB/sample |
| [founder_refit](../methods/refit.md) | Finite: O(N+(S+1)K³+K sum_r Uu_r)/site over S sweeps; worst K⁴/sweep. Non-finite: up to K⁵/round | 4 N L label expansion; O(K²) table/candidate scratch per tile | above cubic: exception 0e ([FR04](../methods/refit.md#fr04)) | K³=27000 base; K⁴=810000 worst/sweep; K⁵=24300000 non-finite/round |
| [pedigree.prepare](../methods/pedigree.md) | O(N B S²) sources; O(N M S) ALT; O(N B(S²+128S)) bridges | 24 N B S²+1.25 evidence+painting+2GiB | ≤ K³ | S31:right weights 7688 bytes/sample/bin |
| [pedigree.screen_pair_bound](../methods/pedigree.md) | O(R² B G³) pair bound G≤S+1; O(C² P) panel; O(D L) exclusion | 16 N(B-1)S² geometry; 4 P/contig; stack streamed | ≤ K³ | N2500 B1000 S31≈38GB geometry; 414M pairs≈1.7GB/contig |
| [pedigree.scores_exposure](../methods/pedigree.md) | O((E/2+T)B S(S+1)/2) exposure; O(N B J²) junctions; O(C N²) order/fit | 8 N B S(S+1)/2 masses; 32 N² outputs; 40 N² family | ≤ K³ | S31:496 features×3123750 unordered edges×B |
| [pedigree.decision](../methods/pedigree.md) | O(C(N²+T)) stack; graph O(N(N+edges)+A) | four 8 C E arrays+likelihood/junction; N² dominant | no K term | N2500 C22 each 8 C E≈1.1GB |
| [pedigree.resampling_release](../methods/pedigree.md) | ≤1000+C fits; O(C A+C N²+I(A+N²)+R A+R N²+R² T+graph)/fit | ≈80 N²/worker | no K term | N2500≈500000000 bytes/fit worker |
| [phase.solve](../methods/phase.md) | O(B(N S³+S² sum A_n))/sweep; O(children B S³) pairing | 8 N S² B emissions; O(B(S+2)²G) pointers | ≤ K³ | N2500 S31 B3000–3300:57.66–63.426GB emissions |
| [phase.release](../methods/phase.md) | O(N B(S+2)³ G) posterior; O(N L S²) site; O(E B) chains | 12 N L spool+5 N L site+E | ≤ K³ | K30 S31 L330000≈3e8 pair visits/sample |
| [recombination](../methods/recombination.md) | O(E M) following; O(E Bm+calls) aggregate; O(loci² E+loci² parents grid) | 16 E Bm+phase chains | no K term | quadratic loci detector separate |
| [sites_export](../methods/export.md) | O(M) scan; O(N M) sample VCF; O(M(C+K)) founder VCF; O(B K²) flags | 2 N L calls; O(M) sites | ≤ K² | N2500 M400000:1e9 cells; calls 2GB |
| [reports](../methods/export.md) | O(N chunks K) painting writer; O(N²) optional pairs | one decoded product+buffers | ≤ K; optional pairs: exemption 0d | N2500 optional 3123750 rows |
| [simulation](../methods/simulation.md) | O(N L) transmissions/reads; linear founders | 2 N L/contig haplotypes; 48 N L reads | ≤ K | N2500 L400000 C22 all-resident 44GB |
| [evaluation](../methods/evaluation.md) | O(sum K L F) counts; O(K³) assignment; O(N L) phase; O(windows F²×100) | calls/truth/mask; no reordered calls copy | ≤ K³ | K30 F≤60; assignment 27000 units |
| [contracts](../methods/core.md) | O(K L) views; O(B K²) partner changes | O(K L) panels; N L founder view | ≤ K² | partner pair factor 900 |
| [core.ties_numerics](../methods/core.md) | O(n log n) sort; O(C n) sum; O(n) expansion | O(n) scratch | no intrinsic K/N | array-size proxy only |
| [kernel_warmup](../methods/execution.md) | O(reachable signatures) compilation; worker parallel | compiler/cache dependent | no intrinsic K/N | compilation of reachable signatures; no intrinsic K |
| [store](../methods/execution.md) | O(bytes) encode/decode; selective items | bounded in-flight payload | no intrinsic K/N | payload-size proxy only |
| [engine_graph](../methods/execution.md) | O(tasks)/boundary; O(contigs nodes) graph | O(tasks + live expansion results + public outputs); geometric ledger growth | no intrinsic K/N | task-count proxy; overhead separate |
| [interface](../methods/execution.md) | Parsing/validation; negligible | O(configuration) | no intrinsic K/N | settings-resolution proxy only (interface.settings); no scientific K ladder |


## Founder refit

For this row S denotes coordinate sweeps and Uu_r denotes uncertified tied candidates with no row shared within their class in two-row round r. Its sum covers applying and releasing rounds. Finite tables use cached flip gains, O(1) joint gains and conflict tests, linear ordering/components, shared-row marking and a certificate over tied-member endpoints. Scans involving shared members total O(K³) per site as their rows are released permanently. The remaining term is K sum_r Uu_r, bounded by O(K⁴) per sweep. All-certified sustained matching classes have zero remaining term; order K uncertified members per round over order K rounds is also cubic. This is a residual worst-case exception, not a universal cubic guarantee. Zero likelihoods at robustness epsilon 0 can produce non-finite tables, which retain the direct route and its O(K⁵) one-round worst case.


## Evidence dictionary windows

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

## Selection at low depth

Small selected K does not imply cheap selection: feedback banks can fit many temporary panels, and one fixed-K fit can require many frequency updates and context snaps. The selected endpoint does not describe all attempted work.
