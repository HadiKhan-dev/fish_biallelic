# Assembly work and memory in 2.7

The instrumentation separates elapsed time from scientific decisions. Join records expose
`phase_seconds` for input gathering, linker, initial beam, join search, the inclusive count step,
and row-path release. Refinement records expose the inclusive count pass, message construction,
window proposals, acceptance, and guarded acceptance. Guarded acceptance is the whole acceptance
pass when the guard is active; it is not an isolated traceback timer. Work counters include score
counts, window/template solves and reuse, programmes, canonical fixing, guard tests and traced
sample-blocks. Final export records carry founder-release elapsed time and work counts.

Times summed across concurrent workers are work attribution, not elapsed stage wall time.
Count time is included in search time; guarded acceptance is included in acceptance. Fit work
includes executed losing branches, while biological count histories still describe the selected
history. Resumed histories contribute only newly executed work. The initial instrumentation-only
profiling snapshot predates that last correction: its selected-fit inner refinement totals are
lower bounds, although its outer component and stage elapsed times cover the whole attempt.
Completion is outside these totals and is removed by the separate 2.7 completion change.

## Exact changes

Join and refinement emission tables store one triangle of unordered founder pairs, indexed by
`max(a,b)*(max(a,b)+1)//2 + min(a,b)`. A block uses
`8*N*K*(K+1)/2*bins` bytes instead of `8*N*K*K*bins`. Mathematical emissions, recursion,
objectives and tie decisions are unchanged. At K30 this removes 48.33% of table storage.
Contiguous sample and site selections view shared evidence directly; discontiguous selections
still gather exactly their selected rows/sites. No claim of zero copies applies to those cases.

The radius-three uncertain-core posterior retains sparse flip sets through prior, marginal and
support/dosage work. It never constructs and scans K dense allele entries for every one of the
O(K^3) support points. The support, its order, ascent/recentring, prior and release thresholds
are unchanged. The small-core exhaustive cube remains capped at 2^10 configurations. The larger
radius rules at core sizes11 and12 remain unchanged; the fixed radius-three benchmark isolates
the large-core path rather than conflating different supports.

Exact join suffix-exchange scores reuse prefix and suffix messages. For each exchanged pair,
only O(K) unordered states move; the best unchanged state is obtained from a sorted list while
skipping at most2K−2 moved states. All pairs cost O(N*B*K^3 + N*m*K^2), with per-worker prefix
scratch rather than a prefix tensor for every sample. The existing screening bounds, candidate
order and plateau-observation callbacks still decide which exact values are consumed. Replacing
those bounds with the exact values would change released plateau families in near-tie cases.
Only the selected exchange is then rescored by the ordinary forward recursion before its
per-sample scores enter count decisions. This adds O(N*m*K^2) work once per selected exchange.
The consistent forward baseline matters: tiny prefix/suffix cancellation differences around zero
can otherwise change effective-carrier ratios in bootstrap draws with no substantive carriers.
Candidate ranking still uses the shared exchange scores; no threshold or count formula changes.

Guard acceptance keeps carried history every16 block boundaries and reconstructs the required
prefix from the preceding checkpoint. This preserves messages, counts and label permutations,
including after accepted suffix exchanges. It reduces resident carried history by approximately
16-fold and adds at most15 block recursions per reconstruction. A bounded rejection fixture
shows a runtime cost as well as a memory benefit; this is not advertised as a speedup. Memory
admission includes full-width guard tracebacks and event matching, since a guard can expand
to the whole component. This scratch is reserved only during guarded acceptance, with the maximum
configured thread team, and released when the pass returns or raises. Directional searches retain
their message, cache and sparse-history reservations while waiting for window proposals; they do
not retain an allowance for inactive traceback scratch. Resident worker/lane memory remains
separately reserved. Completed fit scores and the initial bidirectional score are reused.

The dual sweep computes each sample/bin maximum over candidate rows once, then reuses it
for all background states. Candidate order, maxima and suffix arithmetic are unchanged. Across
all focal founders this removes an O(N*W*b*R*K^3) term when R scales with K; the remaining
sweep kernels are O(N*W*b^2*H^3), H=max(K,R). The reduction reuses existing scratch and allocates
no extra arrays. This bounds one dual sweep, not the data-dependent window programme or number
of windows and fits. Count-shadow decisions also reuse the already computed smaller-panel cover;
there is no second cover scan and no change to shadow factors or the count criterion.

## Polynomial beam proposals (2.8)

Constant transition matrices define a complete Cartesian path family; equal linkage does not
prove equal genotype evidence. Recognition costs O(B G K^2). The beam always proposes the
existing first-covering members, at most 1+sum_b(K_b-1) paths, without materializing the
Cartesian product. Join search proceeds on those representatives and may refine them. Its
certificate remains relative to its stated neighbourhood and proposed pool.

Other endpoint cuts keep at most the endpoint quota, breaking cut-score ties in canonical
prepared-row content order. The forward/backward frontier is at most the sum of endpoint
quotas; the archive is at most the sum of these per-block bounds. Multiplicity overrides
are included in those sums. Path arrays and step rescoring have polynomial block-length
cost. For fixed quotas and B, candidate expansion is quadratic in K; the current per-endpoint
selection scans add a cubic term. No complete tie class can grow a frontier exponentially.
There is no materialization-bound stop or unjoined-input fallback.

This changes proposals inside cut ties and constant families; it is not an exact speedup.
Small nonbinding cut ties can change too. Historical zero beam-stop records in 18,518 cached
joins across six runs demonstrate only that the old stop did not bind, not equivalence of
2.8's proposal pools. Focused tied fixtures and the integrated 2.8 matrix assess those changes.

## Limits of a cubic claim

Two-row exclusion maxima now choose between two exact algorithms using a proportional
work model. Direct forwards cost O(m*P*K^2) for P requested deletion pairs. The segment
programme obtains all requested interval maxima with at most seven scans of the K(K+1)/2
unordered states, then combines complete segment starts. It costs O(m^2*(K^2+P)) per sample;
only mathematically dominated starts are removed. No first-N truncation or computational
abstention is introduced. The original two-region bounds, screening cutoff, candidate domain,
flat-neighbor observer and released-family rules remain unchanged.

The choice uses estimated loop work, not likelihood outcomes. When direct computation is
chosen, its estimated work is no larger than the segment estimate. Both routes therefore
have the same O(m^2*K^2) preparation envelope for one sample/chunk, up to fixed loop constants.
Structural chunks contain complete consecutive first-founder groups and never split a group;
there are at most K chunks. The whole structural-screening pass is consequently bounded by
O(N*m^2*K^3), plus unchanged local-alphabet terms. This is cubic in K **for fixed m**; it is
not uniformly O(N*m*K^3), and it does not bound the number of search, seed-refit, count or replay
passes. The direct route may still show a finite-grid quartic trend before the crossover.
Neither a formal bound nor the adaptive choice is a universal elapsed-time improvement claim.

The segment route's principal scratch is 16*m*Q+17*m*P bytes per active sample, Q=K(K+1)/2,
including emission gathers, interval weights, query histories/scores and live flags. The memory
governor reserves this plus index arrays for each simultaneous numerical thread and each
concurrent seed call. Long-bin cases use direct computation whenever its work estimate is
smaller; they do not allocate the large segment buffers on that route. Floating reassociation
is checked at the actual survivor/flat-neighbor boundary and in released joins, not only by
comparing continuous forward values.

Fine and coarse refinement windows have widths 8 and 4. Their unpruned search spaces grow as
R^8 and R^4 in the local alphabet R. Dominance tests, dual tightening and canonical restricted
programmes add data-dependent work. A cap on stored frontier bytes is not a cubic work bound:
children can be generated and rejected before the retained-frontier check. Finite-grid empirical
exponents must name the input, window width, frontier cap and fallback counts. A capped weak-window
probe does not remeasure the historical whole-L2 K^7.2 result.

## Exact two-row validation

The bounded checks cover 123 complete-pair trajectories at K5–30 and m8–2000 (maximum drift
5.82e-11), 40 structural-bound and survivor/flat-neighbor cases (maximum bound drift4.55e-13),
two saved K30 joins and two fresh full joins. The fresh joins exercise the segment branch and
retain identical released panels and families. All5,495 observed flat panels in the large near-zero
fixture survive. Both row conventions and complete founder-group chunkings are checked.

The adaptive whole structural-pass exponent on K5/10/15/20/25/30 is2.167 at m8 and2.414 at
m128, versus2.317 and2.422 for the original route. K30 is2.27% and3.35% slower in those fixtures.
These are bounded warmed measurements, not a whole-pipeline speed claim.

All six complete standalone two-row comparisons against matching saved2.6 pass: std_n80,
lad_K30_n80, f3_fam, backcross_s6202, plain N-series N80 and TroMau. They cover assembly panels
and released ambiguity, painting, pedigree and support, phase and phase sets, recombination,
and exported founder/sample alleles and missingness. Maximum scientific continuous drift is0,
with no discrete difference, missing record or tolerance exceedance. TroMau covers22contigs and
4,219,539,719 scientific cells. The original backcross reader reached its finite deadline;
a fresh read-only attempt completed the same comparison without changing inference or criteria.

This isolated W3 validation retains founder completion and the2.6 refinement objective.
The integrated candidate requires its own composition checks and final release matrix; these
standalone passes do not establish acceptance of the later W1/F changes.

## N80 cache-placement measurement

One additional full-node N80 attempt used the unchanged exact two-row source, configuration and
76-CPU node class. Only its completed cold run's learned manifest was copied to fresh isolated
caches; no compiled payload or scientific checkpoint was reused. Upfront warm-up compiled all
594 requested overloads with 76 processes in 42.498 seconds, with none skipped. The full released
output comparison against saved 2.6 passed across five contigs and 505,010,355 scientific cells,
with zero discrete differences, missing records or numerical drift.

| Seconds, separate timing scopes | Saved 2.6 | Candidate, cold | Candidate, upfront manifest |
|---|---:|---:|---:|
| Run-record wall, including warm-up/publication | 486.09 | 1028.98 | 447.17 |
| GNU command wall, including CLI startup | 492.69 | 1037.59 | 449.30 |
| Upfront kernel warm-up | 61.080 | 0.010 | 42.498 |
| Assembly summed lane wall | 379.182 | 1122.924 | 367.730 |

The 137.472-second identity comparison is outside pipeline timing. Lane times overlap and must
not be added to main wall. This single intervention supports compilation placement as a major
source of the cold-run slowdown, but compilation inside those cold stages was not timed separately.
The candidate manifest has 594 overloads and no timing priorities; the historical reference used
747 overloads with different coverage/order. Its upfront warm-up is 18.58 seconds longer. The runs
use distinct nodes, not repeated measurements on one node. Neither the total-wall difference nor
the restored nonassembly timings establish a general scientific-kernel speedup.

Sampled process-tree PSS peaks are 75.81 GiB cold and 15.22 GiB with upfront compilation. The latter
has only eight samples about 60 seconds apart: the entire 42.5-second compiler warm-up falls between
the first two samples. Its true compiler-pool peak may therefore be missed; saved 2.6 has no matched
PSS measurement. GNU maximum RSS, governor reservations and available-memory deltas have different
scopes and cannot replace PSS. Short-lived workers may also be absent from sampled CPU totals;
those totals and GNU CPU time do not measure allocation utilization. No continuous peak-memory
reduction or general memory improvement is claimed.

The standalone exact cold runs retain their measured regressions: K30 run-record wall is2685.33
versus2297.10seconds for saved2.6; TroMau is4241.62 versus3863.97seconds. The warm N80 experiment
does not correct those measurements or isolate their kernel time. CPU models and cache histories
must accompany timing comparisons. See [performance](../performance.md) for the integrated release
manifest implementation and its separate cold-cache validation.

The remaining window frontier and the two-row solver's quadratic bin dependence are explicit.
No blanket cubic bound, general elapsed-time improvement or general peak-memory improvement is claimed.

## Whole-component stop for window work

The recalibrated window candidate gives each actual fine or coarse template the fixed account
`B = C N W b² H³`, with `AssemblyConfig.refinement.window_work_coefficient = 33554432`,
`H = max(K, R)`, actual working founder count K, largest actual unit alphabet R, actual unit count W,
full evidence sample count N and largest actual bin count b. Temporary copy founders and every
coarse context path contribute to these dimensions. No dimension is clipped at 30. The coefficient
is the smallest power of two strictly above twice the largest measured normalized charge in
98 original realistic component replays. The factor-two margin was declared before those
inference-only measurements; no truth or output-quality metric selected C. The previous K5-only
coefficient 2^21 is rejected for 2.7 because it fragments TroMau founder identity. The new value is
a computational allowance, not a biological threshold. See the calibration and coverage below.

One account charges every focal, initial programme, dual/restart/canonical solve, dominance scan
and memory retry. Integer scaling rounds credit down and charges up; one final receipt per template
reports raw-unit bounds, scaled units, shape, coefficient and phase work. These normalized loop-domain
charges are not measured FLOPs. They bound per-template charged work cubically in H at fixed N, W and b;
they do not establish a whole-component, whole-count or whole-stage cubic runtime bound. Changing the
fixed retry schedule can change whether a finite account binds; memory-admission pacing preserves that
schedule. The checked integer representation has an explicit supported range and raises on overflow.

A local work stop escapes copy-response preparation, both directions, count/replay and branch ranking.
The complete component refinement/count attempt is discarded, including earlier accepted changes and
completed fits. The exact original rows and count and every incoming full, partial or unresolved family
are retained. Count is reported unverified; there is no partial objective, selected fit, count decision
or convergence claim. Checkpoint reuse is bound to the existing source and configuration identity.

Released continuation metadata carries complete single-focal Cartesian domains for the observed
work-limit witnesses, with the other source rows fixed and original component-forward coordinates.
These domains describe computationally unevaluated searches, not biological ties or equivalent panels.
They do not enumerate every remaining unperformed window, direction, copy seed or component branch.
Concurrent witness lists can depend on worker availability; retained original rows and incoming families
do not. No first-N partial family is accepted, and computational non-completion adds no biological
ambiguity mask. Binding can change downstream scientific output relative to completing the search.

### Cost calibration and bounded weak-data measurement

The exact main/two-row records contain aggregate template/programme counts, not per-template
required work. Two full 76-CPU node replays therefore counted original native component inputs with
an exact-integer account and unchanged focal/retry order. The 871,111 completed template executions
cover 98 components selected solely because the old 2^21 policy bound there. All 98 final row arrays
match their saved native endpoints; none hit the measurement ceiling. This is selected-component
coverage, not every template in every dataset. No W3 exact stress, K20 or reallike arm/per-template
record exists, so no required work is inferred for those datasets. No truth entered calibration.

| Dataset | Components | Template executions | Maximum normalized work |
|---|---:|---:|---:|
| backcross_s6202 | 49 | 348,502 | 5,328,630.575000 |
| f3_fam | 1 | 58 | 4,415,437.418651 |
| lad_K30_n80 | 1 | 3,494 | 3,960,177.211062 |
| n80 | 2 | 25,782 | 3,263,906.927697 |
| std_n80 | 9 | 37,432 | 3,717,526.022400 |
| tromau | 36 | 455,843 | 10,938,598.993534 |

The maximum is TroMau chr15 L3[0,451): N=116, W=8, b=1, K=8, R=7, H=8, charged work 5,197,322,171,392
over denominator 475,136. Twice the normalized maximum is 21,877,197.9871; the smallest power of
two strictly above it is 33,554,432 (2^25), about 3.068 times the measured maximum. The factor-two
margin was declared before these measurements. C is frozen before the confirmation runs.

The bounded weak-window fixture remains N=80, W=4, b=1, seed 36271021, final frontier 512. Two actual
pool workers use one numerical thread each and median times from two warmed repeats. Both arms use
the native template/retry orchestration. The native 2^25 candidate binds only K20; all five unbound
points retain exactly the same proposal rows and existing scientific diagnostics.

| K | Complete normalized work | Main seconds | C2^25 seconds | Work bound |
|---:|---:|---:|---:|---|
| 5 | 1,463,880.076800 | 0.026602 | 0.030213 | inactive; identical |
| 10 | 19,128,145.513600 | 1.301275 | 1.382166 | inactive; identical |
| 15 | 17,120,845.187556 | 3.583192 | 3.630182 | inactive; identical |
| 20 | 37,957,817.189500 | 14.712449 | 13.826519 | binds; no proposals |
| 25 | 17,704,772.743936 | 18.531913 | 19.369954 | inactive; identical |
| 30 | 17,172,731.778370 | 32.613990 | 32.129265 | inactive; identical |

Log-log runtime exponents (K5..30 / K10..30 / K15..30) are 3.905 / 3.008 / 3.036 for main and 3.827 / 2.956 / 3.043 for C = 2^25.
The K20 account avoids at least 11.601% of its complete charged work. C2^26 would bind none of
these six points. This meets the predeclared criterion that at least one K20/25/30 point bind;
it is a limited separation, not evidence that all large-K data bind or that total runtime has
a cubic exponent. This remains one capped window, not the historical whole-L2 K^7.2 workload.
Driver elapsed 206.099 s includes compilation; point timings exclude it. Peak worker
RSS is 1041.7 MiB main and 1632.8 MiB candidate, including JIT and allocator history.

At 2^21 the old upper-range runtime exponent was 3.824 versus 3.039 main. Its TroMau outputs
fragmented 37 released segments into 486 (chr14: 1 into 465), founder columns 308 into 4,372,
and dense export slots 78.5M into 1,082M. Passing the older pooled truth gates did not justify
that loss of long-range founder identity. The rejected coefficient and its outcomes are not
used as acceptance evidence for 2^25.

### Exact alternatives investigated and gate status

Exact focal-emission equivalence classes preserve complete membership and canonical original row IDs.
A prototype matches exhaustive small-window optima and both conventions, including active frontier
caps 1, 4, 16. Classification over all focals costs O(N W b K²R), but ordinary K30 still has 28.25 classes
on average and the reached frontier does not decrease. The distinct-class frontier remains exponential;
this prototype is not included. A same-endpoint prefix ordering reversal under different suffixes
rejects a scalar last-row DP shortcut; a nonadjacent mixed difference rejects a pairwise row-chain
factorization of the shared-row objective. These are specific counterexamples, not an impossibility
theorem for every exact algorithm. Existing dual gaps also prevent declaring the difficult windows
solved from a zero-gap certificate alone.

Full-node confirmations against the completed W3 exact two-row arm pass for TroMau, K30
and backcross_s6202. Backcross had the largest absolute old binding count (49); K30 had
the largest old binding fraction (1/25), so both interpretations are covered.

| Dataset | Contigs | Window bindings / refinement records | Scientific cells compared | Maximum absolute drift |
|---|---:|---:|---:|---:|
| tromau | 22 | 0 / 1155 | 4,219,539,718 | 0 |
| lad_K30_n80 | 5 | 0 / 25 | 34,711,634 | 0 |
| backcross_s6202 | 22 | 0 / 5061 | 6,956,095,641 | 0 |

TroMau retains 37 released segments, 308 founder VCF columns and 78,492,741 dense export slots,
all equal to the exact arm. Chromosome 14 retains one release segment with nine columns.
These are released-export dimensions; the raw assembly census is a different representation.

The comparison covers released assembly components and source families, founder panels and
missingness, painting states/member classes and ambiguity, pedigree calls/support, phase,
recombination and both VCF products. All record/contig/export coverage checks complete.
Scientific tolerances are unchanged: exact discrete decisions; absolute 1e-8 plus relative 1e-12
for ordinary continuous fields, absolute 1e-10 for probabilities/support. The checker-manufactured
default-coefficient leaf (None to 2^25) is retained as an operational difference under the
predeclared contract. No released field or rollback-family semantic is exempted.

These gates apply to the frozen W3 MAIN + exact two-row + recalibrated window source. They do
not validate later integration changes to the objective, completion or scheduling. The selected
cost replays for std, f3 and plain N80 are bounded evidence; no new whole-case confirmation is
claimed for them. The old 2^21 policy stays rejected.

The 2.7 implementation and measurements must be assessed per kernel and per supported path.
This window overlay does not change two-row certification or establish a uniformly cubic
assembly stage. The exact two-row overlay is delivered and validated separately.
