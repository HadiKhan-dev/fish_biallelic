# Founder scaling and assembly models

The default is **dense transitions plus bounded panel search**. The optional
structured transition model gives near-quadratic founder-count scaling. These
choices affect L1–L4 assembly, independently of Stage 1.

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
There is no automatic K cutoff. Both choices use `PanelSearchConfig`, and
both retain the same full Viterbi/BIC acceptance objective and fitting limits.

Python callers can pass `AssemblyConfig(structured_transition_config=None)`
for dense transitions or an explicit `StructuredTransitionConfig()` for the
structured model. The panel-search configuration defaults to
`PanelSearchConfig()` in either case. Explicit `panel_search_config=None`
remains an internal old-search reference for controlled comparisons; it is not
a third CLI mode. Actual configurations are recorded in checkpoint identities.

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
| Candidate paths | Fixed endpoint quota; archive interior-state paths; no MMR all-selected comparisons | O(K² log K) for fixed B/quota |
| Panel selection | Conditional, no-switch and candidate/mate-HMM proposal scores; bounded full Viterbi/BIC refits | O(R (N m K² + B K² log K)) for an O(K) candidate pool |
| Cavity carrier probabilities | Sum each pair-state mass at its one/two founder endpoints | O(N K²), replacing an O(N K³) dense incidence product |

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

### Search changes are approximations

The path pool retains up to 16 paths per endpoint and archives intermediate
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

Full q16 T11/T12 validation has not been run. Completed full-genome final-phase
comparisons use earlier q4 inputs, not q16; see [validation](validation.md).
Stage1, per-level assembly, painting and pedigree checkpoints remain separate.

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
