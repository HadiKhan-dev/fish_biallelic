# Founder scaling and assembly

Assembly has **one route** at every level (L1-L4): dense learned linker
transitions, one dense transition mesh per batch, an endpoint-quota beam over
that mesh, and certified panel search: same-count moves by the join likelihood,
founder counts by the pipeline's effective-carrier rule. There is
no alternative transition model, search breadth or automatic K cutoff. The
additional all-founder refinement within final assembly has separate total-work
terms, some above cubic (below). See
[final refinement](methods.md#final-founder-path-refinement).

The default uses **16 full proposal scores per category**, after the initial
four-score budget caused serious localized reconstruction losses. The wider
budget passed a 22-chromosome assembly/painting comparison, six additional
controls and a perfect metadata-free pedigree check. It remains a heuristic
search: some local outcomes differed from the former broader search, and a
32-score pilot shows a further accuracy opportunity. See the results below.

## Removed assembly options

The structured transition model, the broad assembly search (diversity beam,
founder cap 12, chimera resolution) and panel rehousing were removed; the
batched discovery search can no longer be selected. Their CLI flags no longer
exist; obsolete TOML keys and
`HAPLOTYPES_*` variables are accepted only at the remaining value, with a
warning (see [running](running.md)). `AssemblyConfig` has no transition-model
or search-mode fields; `panel_search_config` holds the base endpoint quota and the tier-1 budgets.
Checkpoint identities record the configuration actually used.

Block discovery remains on its established missing-aware reversible-cavity
search. Painting, pedigree inference, family refinement, phase polishing and
map generation do not depend on assembly settings, though their outputs change
when their input haplotypes change. All four assembly levels share one linker
and retain the 20-iteration transition-fit limit, observation masks, source
provenance, supported component boundaries and dynamic CPU allocation. Panel
selection compares anchor-based initial panels, runs the bounded tier-1 sweep and,
at every stall, the tier-2 certification and pool-saturation steps; its sweep
backstop (100 + 20 max_b K_b) is reported, never expected to bind.

Worker counts per assembly level come from available memory divided by a
per-worker budget: the larger of `min_gb_per_worker` (4 GB) and the largest
batch's estimated arrays (two float32 genotype-probability slices over the
batch's site span, linker scans, binned sub-emissions and their static totals,
the int32 painting and its back pointers bounded by `tensor_budget_mb`, plus the
certified search's exclusion forwards, one chunk of per-sample bounds and the
move and beam lists of the quota-1024 saturation step that a join whose 256
rung accepted a move can reach). In all
current simulations and the real cross (K <= 9, N <= 120) the 4 GB floor sets
the budget.

## Algorithm and complexity

Here K means the largest *working* founder panel, including temporary panels,
not the unknown biological founder count. N is sample count, L is marker count,
B is blocks per assembly batch, m is scoring bins, and I/R are fitting/search
iteration budgets. The normal batch width is fixed at 10. Bounds below are in K,
not claims that sample count, chromosome length, iterations, or I/O are free.

| Component | Approach | Founder-count dependence |
| --- | --- | --- |
| Macro inference (linker) | Arbitrary dense learned transitions; full diploid posterior | O(I N K³) per boundary |
| Candidate paths | One dense mesh per batch; fixed endpoint quota; archive interior-state paths | O(K² log K) for fixed B/quota |
| Panel selection, tier 1 | Conditional, no-switch and candidate/mate-HMM proposal scores; bounded full Viterbi refits of same-count moves | O(B N m K² + R (N m K² + B K² log K)) for an O(K) candidate pool |
| Panel selection, tier 2 (when the count step is quiescent) | Tie-aware exclusion forwards; two-region bound on every same-count move of the neighbourhood; exact scores of survivors; quota-doubling saturation of the pool up to 256 (1024 after a move accepted at 256) | O(N m K (\|C\| + B K²)) per pass, \|C\| = O(q B K) pool paths |
| Join count step (at each tier-1 stall) | Deaths exact from the exclusion forwards; pool births bounded by the two-region kernel, exact (sparse) for survivors; effective-carrier test on the total gain and its sequence part; duplicate-seed births at certified panels | O(N m K \|C\|) per step plus sparse exact scores; at most k0 + 2 sum_b K_b + 2 decisions per super-block |
| Cavity carrier probabilities | Sum each pair-state mass at its one/two founder endpoints | O(N K²), replacing an O(N K³) dense incidence product |
| Refinement sweep (fixed K) | Exact messages; window proposals scored exactly; all-junction founder-pair suffix exchanges (sorted scan) | Per sweep O(N L K²) messages, O(B N K³) suffix exchanges, window programmes and proposal scoring (below) |
| Count decision | Rows without a private cell first; the r = 9 top-ranked one-row deletions (or duplicated-row seeds) refitted to convergence concurrently, all of them when K <= 9; effective-carrier criterion on the total gain and its sequence part | At most 9 converged fits per decision; criterion O(N L K²) incl. one shadow-panel score; novelty O(K² L) |

The hierarchy bounds exclude the later all-founder final refinement, whose
costs are listed below.

The carrier-probability change is mathematically equivalent and also applies
to the standard profile. Homozygotes contribute once to carrier probability.
It changes neither allele likelihoods nor support thresholds.

These are bounded-search algorithms, not polynomial-time guarantees of globally
optimal founder discovery or panel selection. Raising search budgets adds their
cost explicitly. Discovery clustering can remain quadratic in its candidate or
sample cloud; pedigree retains its separate sample-count costs. Missing-allele
completion enumerates all 2^U assignments only for U ≤ 10 unresolved founders
at a site; above that, the uncertain-core site posterior (`assembly/site_posterior.py`)
keeps it polynomial, O(N U³) per site, with no U cap. Block feedback's context
refit uses the same rule over its K context founders (no ten-founder cap).

### Join founder counts

The join (`assembly/panel_count.py`) decides its panel size with the pipeline's
effective-carrier rule (`count_criterion.decide_with_sequence`): a count move
compares a larger panel X+ with X- = X+ minus one path f (a death of f or a birth
of f), and X+ is kept iff the test passes on the per-sample total gain
d_s = V_s(X+) - V_s(X-) and on its sequence part q_s = V_s(X+) - V_s(X+^sigma),
where X+^sigma replaces f's cells that X- lacks by the nearest rows X- uses
(Hamming over called sites). A path without a private cell, or within 1% of
another panel path over the span (near-identical = identical), has q = 0: it is
never added, deleted first, and no same-count move may produce one. Count moves
are single deaths (paths without sequence first, then the best-F candidate) and
single pool births (the largest total gain among the births the rule accepts);
duplicate-seed births split a chimera that covers another founder's cells.
Merges, batch moves and final-L1 pruning are gone: each would introduce a
recombinant of the larger panel. Founder refinement decides its counts with the
same rule (`assembly/founder/converged_sweep.py`): the shadow of a deletion
replaces the decided row's private cells, that of an addition every cell the
refit added that the incumbent lacks; refinement scores the shadow exactly and
takes no near-identity shortcut (its last level is final, so no later level could
re-split a merged pair). Each super-block makes at most
k0 + 2 sum_b K_b + 2 decisions.

### Final founder refinement: explicit work and parallelism

Final founder refinement runs the converged search after each executed level of
final assembly; its count pass adds rows only at the last level. This does not
change the L1/L2 feedback rounds or the hierarchy linker.
See the [model](methods.md#final-founder-path-refinement).

Let K be the working founder count, Q = K(K+1)/2 unordered pair states, N
samples, L markers and B prepared blocks; F is a window-programme frontier, R the
local rows per block offered for one founder and b the bins per window unit.
They are work factors, not biological constants.

| Work unit | Numerical work |
| --- | --- |
| One full score, count or painting (Potts transitions are O(Q) per site) | O(N L K²) |
| Forward/backward messages, once per sweep | O(N L K²) |
| Exact scoring of one proposal on its l markers | O(N l K) for samples whose leading pair state avoids the moved founder, O(l K²) for each other sample |
| All-junction founder-pair suffix exchanges, per sweep (sorted scan) | O(B N (K² log K + K³)) |
| Window programmes for all founders, per sweep (shared background) | O(B N (b² K³ + b K² F R)), plus frontier dominance tests |
| Shared-junction guard tracebacks, per guarded coarse candidate | O(width x affected samples x K²) |
| One count decision | min(9, K) converged refits run concurrently, plus, when K > 9, the ranking: K unrefitted deletion scores O(N L K³), or K seeds' window best responses and copy-only passes O(B N b K² F R + N L K³) |

The number of sweeps per fit is data-dependent (100 is a reported backstop).
Suffix-exchange gains use a sorted scan of the junction terms (the states a
transposition leaves unchanged are read from the per-sample order), with the same
values as the direct scan. Window programmes carry, per partial path, only the K
pair states containing the focal founder and one switch floor per bin; the other
states are shared by all children of a parent. Dominance tests whose outcome is
not clear of the tolerance by a float margin, and kept paths, use the full
messages. A screen before every programme (the window's relaxation bound, every
sample choosing the focal rows freely) skips windows whose incumbent it already
certifies. Canonical ties run a restricted programme only for a row ordered
before the optimum's whose relaxation bound can still reach it; when several such
rows remain, one programme over all of them first checks whether any can. A
programme whose frontier passes 256 paths restarts with its pruning bound raised
to the value of a 16-wide beam of the same programme (a feasible value, so the
optimum and the canonical answer are unchanged). Scoring a proposal in the
acceptance pass advances only the pair states of the moved founder for every
sample whose current leading state avoids that founder and is not overtaken: the
other states then follow the current panel's forward chain exactly; the remaining
samples use every state. A count round refits at most nine candidates (every
candidate when K <= 9), concurrently, so a count step that moves K by d costs at
most 9 (d + 1) refits. Every fit (start fit or refit) is the better of two
concurrent searches, forward-first and reverse-first, which share their common
first sweep's messages and window proposals: at most twice the work per fit, the
same exponent in K.

Window proposals use bins of at least 200 SNPs, targeting 2,000 bins per
component; L1/L2 share a chromosome-derived minimum. Acceptance always uses
individual SNPs.

Independent components run in forkserver worker processes with private evidence
slices and workspaces; each worker's Numba threads are the budget divided by the
active workers, re-read at every sweep, so stragglers take over freed cores.
Inside a component, window programmes (one nogil kernel per window), a fit's two
searches, a round's refits and an addition round's seeds run concurrently, each
with its share of the budget among the work still running. The acceptance pass
is compiled (successive nogil calls); its per-sample and per-candidate kernels
use the search's current share, re-read before every call, and evaluate the
first look of a chunk of junctions at once (chunks double while nothing is
accepted; an acceptance discards the rest). Running kernels cannot resize mid-call.
Scientific selection retains the original candidate order; no decision depends
on the thread count.

### Bounded-search changes are approximations

The path pool retains up to 16 paths per endpoint and archives intermediate
backward refinements. Its size grows linearly with K for fixed batch width,
rather than applying a fixed total founder/path cap. At each stall the pool is
re-generated with the quota multiplied by the number of panel paths sharing an
endpoint, and the saturation step doubles the quota (up to 256, and on to 1024
for a join whose 256 rung accepted a move) before a panel is certified.
Certification is relative to this generator: paths the beam never
proposes, three-way boundary cycles and multi-founder births lie outside the
certified neighbourhood (tier 1 still proposes the latter two).

Proposal scores include exact fixed-painting changes, a no-switch pair
reassignment score, and a small HMM fixing one novel candidate copy while its
mate recombines among current paths. They only rank proposals. Every accepted
same-count panel must improve the join likelihood and leave every changed path
sequence of its own. Coordinated splices and cycle permutations are included
(counts change only through the count step); the permutation
proposal uses quadratic greedy/swap passes, not a cubic assignment solver.

There are at most 20 search sweeps and 16 full refits per proposal category.
This can miss a beneficial change requiring an unproposed repainting or a
different local optimum. Better objective values alone do not establish better
biological reconstruction.

Discovery likewise retains the current missing-aware fitting, wildcard
interpretation, hard-call materialization, and cavity scoring. Only its
candidate exploration changes. It makes no exhaustive-neighbourhood or
calibrated-posterior claim.

## Current 16-score budget: controlled validation

Block discovery and the painting model stayed fixed. The programme completed all 22
contigs of seed402, plus dense seed401 chr2/10/16, dense seed400 chr16/22 and structured seed400
chr16. These are development simulations, not untouched validation seeds.

| Seed402, all 22 chromosomes | Initial q=4 | Current q=16 |
| --- | ---: | ---: |
| L4 closest-founder mismatches | 50,866 | 6,243 |
| Truth-to-panel error/missing burden | 34,102 | 4,719 |
| Called painting alleles / 5,732,385,280 | 5,673,434,537 | 5,727,413,409 |
| Called painting coverage | 98.9716% | 99.9133% |
| Called-genotype error rate | 0.023110% | 0.022917% |
| Switches on the same 493,468,946 eligible pairs | 2,128 | 2,130 |
| Component-aligned allele errors on that common evidence | 50,511,688 | 50,413,352 |
| Sum of L1-L4 chromosome times | 31.15 min | 33.07 min |

The additional 53,978,872 called alleles are not counted as correct merely
because they are called. Raw genotype errors increased from 655,561 to 656,283,
but the called-genotype denominator grew by approximately 27 million. Raw
switches fell from 2,289 to 2,164 on different eligible sets; the matched row
above is the appropriate like-for-like switch comparison.

The full metadata-free q16 pedigree recovered **320/320 exact configurations**:
20 M0 roots, 300 M2 parent pairs and all 600 edges, with no extras or omissions.
Truth was used only for evaluation. That q16 validation run took 28.5 minutes
on 76 cores, including 20.7 minutes in inference. Later implementation timings
are in [performance](performance.md); this older measurement is not the
current runtime or an isolated assembly speedup.

On chr16 across seeds 400–402, q16 gave 1,073 founder mismatches versus 854 in
the broader-search reference, over the same 6,579,146 called founder cells.
Raw painting switches were 438 versus 447. Q16 hierarchy times were 82.6–88.4
seconds versus approximately 152 seconds in the earlier broader-search runs.
Search quality is not uniformly improved by a faster or wider search.

### Remaining search-breadth trade-off

The major seed402 chr14 loss is repaired: q16 recovered six paths with 101
founder mismatches, like the broader reference, and 99.90% painting coverage.
Q8 did not repair it. On chr13, q16 restored the broader reference's called
coverage and reduced mismatches to 679, versus 653 for that reference.

Seed401 chr10 still distinguishes search budgets:

| Budget / search | Founder mismatches | Truth error/missing burden | Raw painting switches | L1-L4 time |
| --- | ---: | ---: | ---: | ---: |
| Broader reference | 5,975 | 12,012 | 65 | 114.1 s |
| q16 | 7,600 | 13,637 | 124 | 68.5 s |
| q32 | 4,115 | 10,153 | 61 | 77.1 s |
| q64 | 4,115 | 10,153 | 61 | 91.7 s |

Q32/64 produced identical chromosome quality and painting metrics on this
case. Their genotype errors were 51,155 versus 50,279 for the broader reference,
with slightly greater called coverage. These raw switch counts have different
eligibility denominators and are not a matched-phase proof. This was a
posthoc-selected failure case: it supports testing 32 more broadly, not silently
assuming that 32 improves every chromosome. Four additional q32 controls (seed402 chr13/14 and seeds400/401 chr16)
subsequently retained the q16 founder-quality and painting metrics, including
the matched phase results. Thus five distinct chromosomes have q32 results,
but a whole-genome q32 pedigree has not been tested. The default remains the broadly
evaluated 16. Python experiments can use
`PanelSearchConfig(full_scores_per_kind=32)` in `AssemblyConfig`.

The later seed403 end-to-end run exercised dense q16 assembly through family phase/recombination
with balanced feedback and final founder refinement. The subsequent multiscale
refinement update changes chr4's founders and has been checked through painting,
but not yet rerun through pedigree–recombination. The earlier q4 comparisons below remain
historical controls; see [current validation coverage](validation.md).
block discovery, per-level assembly, painting and pedigree checkpoints remain separate.

## Removed structured model: historical comparison

Before its removal, the optional structured transition model (sparse specific
edges plus a positive background) was compared with dense transitions. At the
earlier q4 budget, all-22-chromosome seed400 assemblies gave 6,305 (dense)
versus 5,911 (structured) founder mismatches, 2,445 versus 2,465 matched
painting switches on 488,135,746 eligible comparisons, and 31.3 versus 70.0
minutes of L1-L4 time; both metadata-free pedigrees recovered 320/320
configurations. In a four-block K=128 comparison, dense and structured calls
took 42.47 and 60.40 seconds. These results did not establish an accuracy or
runtime advantage that justified a second model, and the combined
batched-discovery experiment had substantially worse phase accuracy.

## Evaluation criteria

Keep these quantities separate when changing model or search:

- closest-truth founder mismatch count and its called-cell denominator;
- truth-to-panel missing/error burden and founder/component representation;
- genotype errors and called coverage;
- phase switches on identical eligible marker pairs;
- component-aligned misphased-allele burden;
- pedigree configurations/edges, final phase and conditional maps;
- time, peak memory and checkpoint scope on the same hardware.

A better full objective or a perfect pedigree alone is insufficient evidence
of better chromosome reconstruction. Missing cells are not counted as correct,
and unsupported components are not assumed to share a phase frame.

Detailed earlier experiment tables and frozen source are preserved in local
ignored storage and the pre-commit development archive. This page describes
the supported choices and current acceptance boundary, not every discarded
prototype.
