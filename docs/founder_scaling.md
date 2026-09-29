# Founder scaling and assembly models

The default is **dense transitions plus bounded panel search**. The optional
structured transition model with bounded search gives near-quadratic
founder-count scaling. An optimized broader search is also available. These
choices affect L1–L4 assembly, independently of block discovery. The default additional
all-founder refinement within final assembly has separate cubic total-work
terms; the structured option does not make that pass near-quadratic. See
[final refinement](methods.md#final-founder-path-refinement).

The default uses **16 full proposal scores per category**, after the initial
four-score budget caused serious localized reconstruction losses. The wider
budget passed a 22-chromosome assembly/painting comparison, six additional
controls and a perfect metadata-free pedigree check. It remains a heuristic
search: some local outcomes differ from the broader reference, and a 32-score
pilot shows a further accuracy opportunity. See the current results below.

## Choosing an assembly model

```bash
# Default: unrestricted dense transition matrices, bounded panel search.
python run.py simulate --config configs/simulation.toml --seed 400 \
  --assembly-model dense --output work/runs/dense_seed_400

# Optional: restricted sparse-specific plus positive-background transitions.
python run.py simulate --config configs/simulation.toml --seed 400 \
  --assembly-model structured --output work/runs/structured_seed_400
```

The same flag works for `astcal` and `tropheops`. TOML uses
`[run].assembly_model = "dense"` or `"structured"`; the environment variable is
`HAPLOTYPES_ASSEMBLY_MODEL`. Precedence is CLI > TOML > environment > dense.
There is no automatic K cutoff. Both transition models default to
`--assembly-search bounded` and retain the same full Viterbi/BIC acceptance
objective and linker fitting limits.

Use `--assembly-search broad` for the optimized broader diversity-beam and
full-refit panel/chimera search. For the earlier dense/broad combination:

```bash
python run.py simulate --config configs/simulation.toml --seed 400 \
  --assembly-model dense --assembly-search broad \
  --output work/runs/broad_seed_400
```

Search precedence is CLI > `[run].assembly_search` >
`HAPLOTYPES_ASSEMBLY_SEARCH` > `bounded`. Broad search retains Numba kernels,
cached swap templates, chunked scoring and dynamic thread allocation; it
does not reinstate unoptimized historical code. It retains the diversity beam
(width 200), founder cap 12, top-20 swap candidates and at most 10 chimera
resolution iterations. It is broader, not exhaustive or guaranteed to improve
allele accuracy.

Python callers can pass `AssemblyConfig(structured_transition_config=None)`
for dense transitions or an explicit `StructuredTransitionConfig()` for the
structured model. The panel-search default follows the search environment
setting; explicit `panel_search_config=PanelSearchConfig()` selects bounded
search and `panel_search_config=None` selects broad search regardless of that
setting. Actual configurations are recorded in checkpoint identities.

Block discovery remains on its established missing-aware reversible-cavity search.
Its independent experimental option is `--discovery-search batched`
(`[run].discovery_search`, `HAPLOTYPES_DISCOVERY_SEARCH`; default `standard`).
The earlier combined discovery/assembly experiment had substantial accuracy
regressions. It is **not** necessary to change discovery to use structured
assembly.

These assembly options do not change the scientific models used by painting,
pedigree inference, family refinement, phase polishing or map generation. All four assembly levels still share one linker and
retain the 20-iteration transition-fit limit, observation masks, source provenance,
supported component boundaries, and existing dynamic CPU allocation. Bounded
panel selection now allows up to 100 sweeps, compares anchor-based initial
panels, and fully rescores both single and coordinated pruning proposals. Downstream stages
can nevertheless change when their input haplotypes change.

## Algorithm and complexity

Here K means the largest *working* founder panel, including temporary panels,
not the unknown biological founder count. N is sample count, L is marker count,
B is blocks per assembly batch, m is scoring bins, and I/R are fitting/search
iteration budgets. The normal batch width is fixed at 10. Bounds below are in K,
not claims that sample count, chromosome length, iterations, or I/O are free.

| Component | Approach | Founder-count dependence |
| --- | --- | --- |
| Optional batched discovery | Batch births/pruning, residual replacements, fixed-start gauge moves; same fixed-K fitter and mean-field cavity score | O(R I N L K²), plus candidate generation/clustering costs |
| Default dense macro inference | Arbitrary dense learned transitions; full diploid posterior | O(I N K³) per boundary |
| Optional structured macro inference | Sparse specific edges plus a shared positive background; full diploid posterior | O(I N K² log K) per boundary |
| Bounded-search candidate paths | Fixed endpoint quota; archive interior-state paths; no MMR all-selected comparisons | O(K² log K) for fixed B/quota |
| Bounded panel selection | Conditional, no-switch and candidate/mate-HMM proposal scores; bounded full Viterbi/BIC refits | O(B N m K² + R (N m K² + B K² log K)) for an O(K) candidate pool |
| Final-L1 allele-preserving pruning | Direct drops and single-recipient local-piece moves; at most Q=256 full scores per count-down round | O(K² L + B K³ + Q N m K³), plus O(N m K² H) if local dictionaries have H rows; cubic when H=O(K) |
| Cavity carrier probabilities | Sum each pair-state mass at its one/two founder endpoints | O(N K²), replacing an O(N K³) dense incidence product |
| Refinement global escape | Budgeted sparse coordinates plus conditional dual search; shared background | O(R D N M [K³(s+1) + C K(K+s)]), plus full-site scoring |
| Count deletion/refit | Sparse repairs for all K deletions, full endpoint choice; one deep refit | Cubic for A=O(K), plus O(N L A³) bound |
| Post-count polishing | Sparse coordinate epochs and verified fixed-point reuse | O(S K²) per sparse painting plus full endpoints |
| Paired suffix/tract search | Exact bounded suffix queries; feasible-tract shortlist with full acceptance | Suffix worst case O(N B (K³ + K² log K)); tract shortlist O(N B K² + B H K²) |

The near-quadratic/cubic hierarchy comparisons assume bounded search and exclude
the later all-founder final refinement. Its per-focal diploid-state update is
quadratic, but sweeping all K founders adds another factor of K.
Broad search restores additional full rescoring and coordinated suffix
proposals, including the historical quartic-style search costs. Selecting
structured transitions alone does not bound that broader search quadratically.

The carrier-probability change is mathematically equivalent and also applies
to the standard profile. Homozygotes contribute once to carrier probability.
It changes neither allele likelihoods nor support thresholds.

These are bounded-search algorithms, not polynomial-time guarantees of globally
optimal founder discovery or panel selection. Raising search budgets adds their
cost explicitly. Discovery clustering can remain quadratic in its candidate or
sample cloud; pedigree retains its separate sample-count costs. Missing-allele
joint enumeration still has a 2^U factor, with U capped at six unresolved
founders per site in the supported completion route. It does not become cheap
if that cap is allowed to grow with K.

### Final-L1 allele-preserving pruning

After selecting final assembly's L1 panels, `assembly/panel_pruning.py` adds
single-recipient moves: retain a deleted row's unique local sequence pieces
by moving them to one survivor whose displaced pieces remain represented.
Direct deletions must retain every represented called allele. Unknown alleles
do not count as evidence. Every accepted panel strictly improves the existing
full Viterbi/complexity score. These are search constraints, not evidence that
each row represents an original biological founder, and not a target row count.

The default scores at most 256 raw proposal descriptions per count-down round.
This exhausts the original matched real/simulation controls (K at most 15/7).
Larger candidate sets are ranked by fixed-painting edit gains before full
rescoring; these are heuristics, not safe bounds. Truncation and evaluated
counts are saved. At most K-1 accepted reductions keep total work cubic for
fixed budget and H=O(K), where H is local dictionary size. Row multiplicity
and deterministic proposal order are retained. The pass runs only at L1 of
final assembly (`max_level=4`), not in either feedback round or at L2–L4.
`AssemblyConfig(l1_pruning_full_scores=0)` disables it for controlled comparisons;
the value is recorded in the scientific checkpoint identity.

### Final founder refinement: explicit work and parallelism

The default is budgeted sparse-coordinate/global search after each executed
level of final assembly, plus separate final count-up. This does not change
the L1/L2 feedback rounds or the dense/structured hierarchy linker.
See the [model and accepted tradeoffs](methods.md#final-founder-path-refinement).

Let K be working founder count, A the largest local/macro candidate alphabet,
N samples, L markers and B prepared blocks. M counts predictive proposal bins,
s is the largest bins-per-block, and S counts sample-specific anchor cells
summed over samples. S is generally much smaller than N*L, but has that worst
case. R<=20 is the shared local/global iteration allowance, D<=20 dual sweeps,
C=min(A,16), J=3 deletion sweeps, Q=16 tract checks, H=100 bounded interval span
(in local blocks or original L1 groups). They are work factors, not free
biological constants.

| Work unit | Numerical work |
| --- | --- |
| Fixed evidence tables | O(N L A²) |
| One full primary score, count or painting | O(N L K²) |
| Anchored sparse painting | O(S K²) |
| Fixed-painting local coordinate sweep | O(S (K+A)); blocks parallel, founders sequential inside a block |
| Rebuild anchored emissions | O(N L A²) worst case; unchanged unsplit cells reuse totals |
| One all-founder global dual round | O(N M K² s + D N M [K³ + C K(K+s)]), plus canonical checks |
| All K cheap deletion endpoints | O(J N B K³ + N L K³) for one 200-SNP ranking bin per local block |
| Optimistic count bound | O(N L A³) worst case |
| Suffix message preparation | O(N L K²) |
| Bounded suffix queries | O(N B K² log K) preparation; at most O(N B K³) exact queries |
| Feasible tract shortlist | O(N B K² + B H K²), plus sorting and at most Q full/local checks |
| Exact local edit over ell markers | O(N ell K²), after O(N L K²) unchanged-flank preparation |

All K deletion starts receive cheap repair; only the best gets deep search.
Count-up uses the same repair for incumbent and K+1, capped at two additions.
Full category checking initially uses four ranked proposals but exhausts a
stalled category, so the worst-case all-founder full-scoring term remains
O(N L K³). Four here is a refinement budget, **not** the hierarchy's 16 scores.

With A=O(K), fixed bin/beam/iteration budgets and fixed hierarchy depth, total
founder-count work remains cubic (up to sorting factors). It is not globally
quadratic, and neither the L dependence nor the block t² background term
disappears. Sparse coordinates improve constants and typical site dependence;
dense/global escape and count work still matter.

Three different resolutions must not be confused:

- Ranking tables use 200 SNPs per bin.
- The predictive proposal HMM targets 2,000 bins, with at least 200 SNPs per
  bin. Keeping local boundaries can exceed the target.
- Count-up seed catalogues group about 2,000 SNPs into a founder fragment,
  while preserving all internal sample-HMM emission bins.

Primary acceptance always uses individual SNPs. Anchored grids retain all
incumbent switches and sum likelihoods; other panels get a lower bound, not
a safe exclusion bound. Feasible tract shortlisting and finite candidate
budgets can change the result. The exact optimizations—immutable table reuse,
cached identical queries, incremental messages, deferred losing counts and
neutral score/count contraction—do not change the intended objective.

Memory is dominated by full evidence O(N L), prepared emissions O(N M A²),
anchored tables O(S A²), and optional flanks O(N B K²). Dosage arrays add
O(L K²). Memory admission can select direct equivalent kernels or reduce
concurrent workers. Checkpoints store original-row paths and non-derived
metadata, not repeated chromosome arrays.

Independent components and candidate tasks run concurrently with private mutable
state and shared immutable evidence. The allocator reserves reusable worker
slots while a queue exists; completed slots redistribute their threads to
stragglers at safe numerical boundaries. Running kernels cannot resize mid-call.
Scientific selection retains the original candidate order.

For the documented AulStu chr3 refinement-only workload, measured contributions
were 0.307 L3 + 0.330 L4 + 0.305 final count-up = **0.943 equivalent 76-core
node-hours**. The stage timings exclude loading/evaluation/hierarchy and do not
predict identical wall time on another CPU generation or dataset. The
[methods record](methods.md#measured-tradeoff) reports substantial real-path
changes and mixed omitted-founder recovery alongside intact controls.

### Structured boundaries are a model change

For a rectangular K-left by K-right boundary,

`T = S + u qᵀ`, with `sum(q)=1` and `sum_j S_ij + u_i = 1`.

S has at most max(2, ceil(log2 K-right)) selected entries per row, bounded by
K-right. Specific edges are screened using independent carrier-profile
associations. The shared background gives every destination positive mass;
states are not hard-pruned and the diploid posterior is not factorized.

Both forward propagation and its true adjoint exploit this structure. Expected
specific-edge counts and background source/destination counts are exact for
this restricted model. Ordinary and extreme-value log-domain kernels have the
same near-quadratic bound; neither falls back to a cubic contraction.

The restricted parameterization and its latent-mixture regularization are
**not equivalent to arbitrary dense learned transitions**. Specific-edge support
is fixed within a fit; a genuinely complex boundary may not be represented as
well. Backward mesh summaries are reverse conditionals derived from the
regularized joint mass, not the adjoint likelihood operator.

### Bounded-search changes are approximations

The bounded-search path pool retains up to 16 paths per endpoint and archives intermediate
backward refinements. Its size grows linearly with K for fixed batch width,
rather than applying a fixed total founder/path cap.

Proposal scores include exact fixed-painting changes, a no-switch pair
reassignment score, and a small HMM fixing one novel candidate copy while its
mate recombines among current paths. They only rank proposals. Every accepted
panel must improve the existing full Viterbi/BIC objective. Coordinated splices,
cycle permutations, and batch births/deaths are included; the permutation
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

## Interpreting the structured alternative

The initial controlled assembly-only comparison froze block discovery and the painter
on chr16 across seeds400–402. Broader dense search versus bounded structured
search gave **854 versus 878** closest-founder mismatches across 6,579,146
called founder cells. Because both the transition model and search changed,
this does not isolate the effect of the structured parameterization.

At that earlier budget, full q4 dense and structured seed400 assemblies were
also compared across all 22 chromosomes. Founder mismatches were 6,305 versus
5,911, but matched painting switches were 2,445 versus 2,465 on the same
488,135,746 eligible comparisons. L1–L4 chromosome times summed to 31.3 versus
70.0 minutes. Both metadata-free pedigrees recovered 320/320 configurations.
These are q4 results, not a completed all-chromosome q16 structured comparison,
and do not establish uniform accuracy or runtime superiority.

The initial combined discovery-and-assembly experiment had substantially worse
phase accuracy. It is not a reason to reject the isolated assembly option, but
it is why the separate batched discovery search remains experimental. Sparse
transitions do not require choosing that discovery search.

At low K, dense kernels and their simpler bookkeeping can be faster. In a
later fixed-search four-block K=128 comparison, optimized dense and structured
calls took 42.47 and 60.40 seconds respectively. Near-quadratic scaling is an
option for large working panels, not a guaranteed speedup at every K.
There is no automatic model switch.

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
