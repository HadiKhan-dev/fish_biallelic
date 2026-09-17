# Founder scaling and assembly models

The default is **dense transitions plus bounded panel search**. The optional
structured transition model with bounded search gives near-quadratic
founder-count scaling. An optimized broader search is also available. These
choices affect L1–L4 assembly, independently of Stage 1. The default additional
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

Stage 1 remains on its established missing-aware reversible-cavity search.
Its independent experimental option is `--discovery-search batched`
(`[run].discovery_search`, `HAPLOTYPES_DISCOVERY_SEARCH`; default `standard`).
The earlier combined discovery/assembly experiment had substantial accuracy
regressions. It is **not** necessary to change discovery to use structured
assembly.

These assembly options do not change the scientific models used by painting,
pedigree inference, family refinement, phase polishing or map generation. All four assembly levels still share one linker and
retain the 20-iteration limit, observation masks, source provenance, supported
component boundaries, and existing dynamic CPU allocation. Downstream stages
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
| Bounded panel selection | Conditional, no-switch and candidate/mate-HMM proposal scores; bounded full Viterbi/BIC refits | O(R (N m K² + B K² log K)) for an O(K) candidate pool |
| Cavity carrier probabilities | Sum each pair-state mass at its one/two founder endpoints | O(N K²), replacing an O(N K³) dense incidence product |
| Final all-founder dual escape | Exact shared-background scans; candidate-specific states only contain the focal founder | O(R D N M [K³(s+1) + C K(K+s)]), plus full-site scoring |
| Count deletion/refit | All K deletions get cheap conditional repairs; only the best repaired panel gets a deep refit | O(J N L (K³+K A)) repairs, one deep search, and O(N L A³) bound |
| Completed window searches | Shared-background candidate scoring, safe optimistic pruning, and proven-equivalent trajectory reuse | Cubic in K for A=O(K) and fixed bin/beam budgets; details below |
| Paired suffix/interval search | Exact suffix relabeling; at most p K expensive interval pairs | Suffix queries O(N B (K³ + K² log K)); interval work O(p N H M K³) |

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

### Final founder refinement: explicit work and parallelism

The complete refiner runs after each executed level of **final L1–L4 assembly**,
not in the two local-feedback contexts. For a fixed four-level hierarchy,
summing work over disjoint components changes constants, not the founder-count
exponent. Primary score and founder-count cost remain fixed; partial-founder
evidence breaks exact primary ties as detailed in the
[refinement model](methods.md#partial-evidence-at-primary-ties).

Let A be the largest local or macro candidate alphabet, M the number of proposal
bins, and s the largest number of bins inside one local or macro block. Let
C=min(A,16), W be beam width (64 initially, up to 1024), R<=20 outer iterations,
D<=20 dual sweeps, J=3 deletion-repair sweeps, p=3 interval partners per founder,
and H be the interval span in prepared blocks. The configured window is 100;
macro grouping can enlarge its span in original blocks. These are explicit
work factors, not quantities assumed to be free.

A focal founder changes only K of the O(K²) unordered diploid states. All
other state emissions are shared across its local candidate choices. For a
block with t bins, the exact max-plus kernel precomputes uninterrupted
background-segment scores in O(K² t²). Each candidate then costs O(K t+t²),
rather than another O(K² t) scan. Entries into the shared background include
arbitrarily many switches: this is not a frozen-background approximation.
Selected beam paths still materialize their full diploid states.

| Work unit | Serial numerical work |
| --- | --- |
| Prepare local binned evidence | O(N L A²) |
| Optimistic founder-count bound, worst case | O(N L A³) |
| One complete-panel score or painting | O(N L K²) |
| One secondary partial-founder tie score | O(N L K²), with O(N P + L K²) extra storage for P partial sites |
| One focal beam round | O(N M [K²(s+W) + W C(K+s)]), plus O(B W C log(W C)) sorting |
| One focal dual search | O(D N M [K²(s+1) + C(K+s)]) |
| All-founder dual round, shared preparation enabled | O(N M K² s + D N M [K³ + C K(K+s)] + N L K³) |
| All-founder window round, shared preparation enabled | O(N M [K² s + W K³ + W C K(K+s)] + N L K³), plus beam sorting |
| All K cheap deletion repairs | O(J N L (K³+K A)) |
| Paired suffix queries | O(N L K² + N B (K³ + K² log K)) |
| At most p K interval scans | O(p N H M K³), plus O(p N L K³) full-site checks |

Multiply iterative searches by their actual iteration counts. With A=O(K)
and fixed s, W, R, D, J, p and H, total work is cubic in K, up to sorting
factors. In particular, cubic scaling no longer depends on hiding an
all-deletion deep-refit factor behind a fixed cap. Bin count is
M=sum_b ceil(L_b / bin_size), not strictly 2,000; it can approach 2,000+B.
The t² block-preparation term is real: long macro blocks can limit practical
speed even when the K exponent is improved. L1/L2 share a chromosome-derived
minimum proposal-bin size, avoiding a separate 2,000-bin budget for every
small component; late levels retain their original component resolution.

Primary-preserving fallback searches use the same bounded dual kernel, at most
two directions per focal founder in an existing outer iteration. Equivalence
groups require O(L A) preparation; caps and incumbent retention remain explicit.
Full-site secondary checks add at most O(N L K³) per all-founder round, not
quartic or quintic candidate rescanning. Unknown source identity is retained,
and complete input requires no additional secondary evidence scan.

**Invariant background reuse:** the production dual/window search prepares
prefix/segment summaries once per fixed competing panel, not once per sweep
or overlapping window. For each sample/bin interval, the best diploid state
already gives the exclusion maximum for every focal founder except its one/two
endpoints. Only those endpoints need another scan. All-founder preparation
therefore costs O(N K² sum_b t_b²), rather than O(N K³ sum_b t_b²) on every
sweep. Message updates, candidate paths and previous-state scores are not
shared. These optimizations leave other cubic terms intact.

The shared tables use O(N [K²(M+B) + K sum_b(t_b+1)²]) memory per panel,
with four scan geometries. They are read-only across candidate threads.
An estimate exceeding one eighth of available process memory selects the
direct, mathematically identical kernels, retaining the preceding cubic
bound without the shared-preparation reduction.

**Localized full-site scoring:** unchanged-panel forward/backward messages
at original block boundaries cost O(N L K²) to build, with O(N B K²) storage.
An edit spanning ell SNPs can then be scored in O(N ell K²), rather than
O(N L K²). Only edits covering at most half the chromosome use this route,
and memory limits can disable the cache. A panel change invalidates its
messages; differing founder counts never reuse them. Possible winners and
near ties receive canonical full rescoring before selection. Final acceptance
and macro genotype-fit guards therefore still use the original full-site
scorer. This is not imputation, a frozen sample painting, or a local-only
acceptance objective.

**Count selection:** every deletion receives up to J repaint/conditional-row
repair sweeps. Each sweep scores the simultaneous row update and the
highest-gain single-row update, not K separately rescored updates. All repaired
panels compete against the incumbent; only the best repaired deletion enters
the expensive beam/dual search. A paired exchange can trigger one further
deep refit of that same candidate. This is a search approximation: a
second-ranked cheap repair could have led to a better deep optimum. The default
also skips that deep refit when every repaired deletion still loses by more
than eight per-founder complexity costs. This is a heuristic budget screen,
not a rigorous likelihood bound; `count_refit_deficit_multiple=None` disables it.
Neither improving repairs nor the initial optimistic bound are altered.

**Intervals:** all pairs are retained when their number is at most p K
(in particular K<=7 at the default p=3). At larger K, existing suffix scores
rank partners for each founder; the union of its best p partners contains at
most p K pairs. Every retained interval still receives exact binned and
full-site checks. Shortlisting can miss an interval whose endpoints jointly
help despite an unpromising suffix score.

**Windows:** a sample-relaxed optimistic bound may prune windows that cannot
beat the best proposal already found. The same bound can abort a partially
expanded window when none of its remaining beam branches can win; no individual
candidate is discarded merely for a weak heuristic score. A corrected dual
bound can also certify that another coordinate sweep cannot improve the best
feasible path, using a conservative numerical margin. Incumbent/tie trajectories are reused
only when their retained branch orders provably agree on the same input
rows; macro trajectories are excluded from this reuse. Exact flank scores
select a winner before one canonical verification. This avoids full
chromosome rescanning after every improving window. Floating-point tie
ordering may change; accepted outputs are checked scientifically, not for
bitwise intermediate equality.

Independent components share immutable chromosome evidence and divide the
verified CPU budget, with private component workspaces and bounded score caches.
Small components checkpoint completion; large ones also checkpoint inner
searches. Completion order does not alter genomic output order.

Within a component, cheap deletion repairs
and deep focal/direction searches run in separately scheduled thread teams
with subdivided, dynamically reallocated thread budgets. Native repair kernels
release the GIL, each repair owns its painting/score workspace, and completed
repairs are checkpointed on the controller before canonical-order selection. Running native
kernels cannot acquire threads mid-call. Memory limits may reduce concurrency,
and explicit Numba workqueue configurations retain serial inner searches.

The current implementation packs selected-panel interval emissions into
O(N M K²) ordinary-array storage and reuses its O(N B K²) flanks across grids;
this removes shared typed-list bookkeeping from the hot scan, not a work term.
Binned full checks and fixed-panel suffixes use contiguous emission buffers
with scalar indexing, packed once per model workspace and released with it.
Retained beam/window states alternate reusable thread-local destination buffers.
These changes reduce native ownership/allocation overhead without removing bins.
Where only a score and switch count are required, the exact first-argmax/strict-
switch recurrence carries O(K²) values per active sample thread instead of a
length-L traceback. Neutral and missing-site conventions are unchanged.

Prepared evidence/logs/masks are shared read-only across deletion workers.
Shallow checkpoints contain O(K B) row selections and scalar scores; only the
selected contenders are reconstructed. Model/code checkpoint identities include
the packing and compact-checkpoint helpers. Intermediate completed components
store local-row paths plus non-derived metadata; loading reconstructs the same
called/inference arrays and boundaries from identity-bound prepared inputs.
All resume tokens remain, and final release products keep ordinary block arrays.
Small dual coordinate updates cap their team at eight,
while suffix and beam kernels retain their complete dynamic allocation.
Independent queries remain separately scheduled, not co-batched into one solver.
For one/two-bin leaves, exact beam scores use direct background stay/enter
maxima and K affected states per candidate; larger macro blocks retain the
generic sparse recurrence. Sample-major scratch avoids strided parallel writes.
Stable partial selection replaces full branch sorting only at >=4096 branches;
original indices resolve cutoff ties. The search width and alphabet do not change.

Suffix exchanges stop canonical rescoring only after a guard-passing winner
exceeds the remaining exact flank scores by a conservative accumulated float64
error bound. Near ties and potentially winning candidates still receive the
canonical score and genotype-fit guard. Proposal binning is unchanged.

These changes improve measured constants without altering cubic total-work
bounds. Those measurements used the earlier final-only schedule. Cross-query dual batching was
tested but not retained: the existing dynamically scheduled queries were faster
on the N80 controls.

Evidence/models require O(N L + N M A²) space per component workspace.
Shared-background scratch includes O(K² s+s²) per active sample, and candidate
rows/entries add O(C(K+s)); beam, message and painting traceback storage remain
additional. Painting traceback grows as O(L K²) per active sample thread.
Parallelism changes elapsed time, not the work exponents. See
[measured refinement performance](performance.md#final-founder-refinement).

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

Stage 1 and the painting model stayed fixed. The programme completed all 22
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

| Budget / search | Founder mismatches | Truth error/missing burden | Raw T09 switches | L1-L4 time |
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

The later seed403 end-to-end run exercised dense q16 assembly through T11/T12
with balanced feedback and final founder refinement. The subsequent multiscale
refinement update changes chr4's founders and has been checked through T09,
but not yet rerun through T10–T12. The earlier q4 comparisons below remain
historical controls; see [current validation coverage](validation.md).
Stage 1, per-level assembly, painting and pedigree checkpoints remain separate.

## Interpreting the structured alternative

The initial controlled assembly-only comparison froze Stage1 and the painter
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
