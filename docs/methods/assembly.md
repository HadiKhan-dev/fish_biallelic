# Assembly

Assembly turns the path-selected local founder panels of one contig (one small panel per discovery
block, each with K_b local founder rows) into chromosome-scale founder haplotypes, split only into
phase components whose boundaries the data cannot bridge. It runs in three steps. Founder
completion fixes the prepared panels: the local calls plus the unknown alleles the data determine
well enough to publish. A hierarchy of levels then joins batches of adjacent components: a
distance-aware linker fits founder transitions between the blocks of a batch, an endpoint beam
enumerates candidate chromosome paths, and a certified panel search chooses the batch's founder
paths, with their number decided by the pipeline's founder-count rule. In the contig's final
assembly, [founder refinement](founder_refinement.md) re-chooses every component's founders after
every level. The same computation has two roles: the final assembly (`Role.FINAL`), the stage
product that [painting](painting.md) reads, and the context assemblies of local selection's
feedback rounds (`Role.CONTEXT`, through level 1 or 2 and without refinement), whose components
seed the next round of [local selection](local_selection.md). No pedigree, cohort label,
simulation truth or target founder count enters.

## Inputs and outputs

| | Final assembly | Context assembly |
|---|---|---|
| Local panels | the `local_selection/l2` panels (`LocalPanel`, one per informative block) | the panels of the calling round |
| Levels | until the stop rule ends the loop ([hierarchy](#hierarchy)) | 1 (round `l1`) or 2 (round `l2`), `LocalSelectionConfig.context_levels = (1, 2)` |
| Founder refinement | after every level | none |
| Records | under `assembly/` | under `local_selection/l1/assembly` and `local_selection/l2/assembly` |

Every assembly also reads the contig's evidence view and genetic map. The evidence view
(`assembly/evidence.py`, built once per contig from the evidence record's founder view) holds the
normalised genotype likelihoods where a sample is observed at a discovery site and exactly 1/3 at
every other cell (unobserved, not a discovery site, or a sample the contamination QC excluded from
founder construction). Completion and the batch partition read it in float64; the linker, the joins
and founder refinement read one float32 copy of the same values. The genetic map gives the
linker's within-block switch probabilities and the beam's maximum gap: distances come from the map
file where one covers the contig (with the fallback rate `[genetic_map] rate_cm_per_mb` beyond its
ends) and from the fallback rate alone otherwise.

The product of the final assembly is the record `assembly/final`: the phase components
(`FounderComponent`, with their count calls) over the prepared panels of `assembly/completion`.
Other stages and the evaluator read it through `assembly/records.py`: `load_product`,
`founder_panels(file, level)` (level k: the level-k product after refinement, or the hierarchy
output where refinement did not run; no level: the final product), `level_records` and
`join_searches` (the reports' assembly search table). `model.components.materialise` turns
components into chromosome-scale arrays. Two report tables are written with each contig's painting
outputs ([painting](painting.md)): `assembly_search.tsv` (every record of the final assembly's join
searches) and `founder_counts.tsv` (how firmly each component's founder count is determined).

None of the stage's constants is a TOML key (the generated [configuration reference](../configuration.md)
lists them). The user options that affect assembly are `[genetic_map] file` and
`[genetic_map] rate_cm_per_mb` (distances), `[quality] sample_filter` (which samples are uniform in
the founder view), `[execution] batch_queue` (hierarchy batches run by helper workers on other
nodes) and `[execution] cores` and `[execution] memory_limit`, which change timing only.

## Prepared panels and founder components

Assembly never copies founder alleles into chromosome arrays. Its atomic unit is the prepared panel
of one block (`PreparedPanel`, `model/components.py`), R local rows over the block's sites:
`inference` int8[R, L] holds the local-selection calls with -1 at non-discovery sites (the frozen
pre-fill snapshot that every likelihood reads), `released` the same calls plus the released
completion fills, and `alt_probability` and `support` the per-allele probability and directional
supporters. Every product of assembly is a row-path matrix: `rows[f, b]` is the local row of
prepared panel b that founder f copies. A `FounderComponent(start, stop, rows, break_before,
break_after, count_call)` covers the prepared panels [start, stop); the components of one assembly
partition the prepared panels in genomic order, and founder labels are row indices 0 .. K-1 of a
component (component-local, with no key objects). `materialise` gives a `ComponentPanel` per
component: positions, released and inference calls int8[K, L], discovery sites, ALT probabilities,
supporters, the site offsets of the prepared spans and `source_rows` (two founders that copy the
same prepared row at a site share one latent allele there). A `CountCall` records a component's
last count step (founder refinement at the final level): K before and after, each decision (kind,
K, kept or added, skipped by a bound, margin, bootstrap band, ambiguity), count cycles and the
overall ambiguity; components with one block or one founder carry the reason
`single_block_or_founder` instead. Stage inputs are never modified: completion returns prepared
panels without touching the local panels, and phase breaks are carried on fresh components.

## Founder completion

Local selection leaves some founder alleles unreleased. Completion (`assembly/completion/`)
decides, block by block, which of them the data determine well enough to publish, without letting
a fill feed back into its own evidence. In the observation model of the cross-fit and the cavity
fill, a sample copies an unordered pair (a, b) of the block's K rows (K(K+1)/2 pair states), its
genotype dosage is x_a + x_b, a homozygous pair copies one allele, an unresolved allele is a binary
variable shared by every sample, and evidence enters through the robust mixture 0.99 P(g) + 0.01/3
(`AssemblyConfig.completion.crossfit.uniform_mix = 0.01`), whose log the pair-state scores floor at
`AssemblyConfig.completion.crossfit.log_floor = -2.0`.

1. **Cross-fitted carrier profiles** (`assembly/completion/crossfit.py`). The samples are split into two deterministic
   folds (`AssemblyConfig.completion.crossfit.fold_seed = 1729`): ordered by a seeded SHA-256 of their observation
   pattern on the contig and alternated (release 2.2), so the split does not depend on sample names, input order or
   the convention reversal, and genotype-likelihood noise cannot move it. On one fold a variational EM over
   per-sample pair states and per-site configurations of the unresolved alleles runs from one
   start (uniform configurations, pair posteriors from the complete sites) until the evidence
   lower bound converges (at most `AssemblyConfig.completion.crossfit.max_iterations = 100`
   iterations). It releases conservative calls: marginal at least
   `AssemblyConfig.completion.crossfit.minimum_call_probability = 0.99`, carrier support (the
   larger of the number of samples carrying the row with probability above 1/2 and their summed
   carrier probability) at least `AssemblyConfig.completion.crossfit.minimum_carriers = 2`, a row
   not identical to another row, no multimodal site and no shell-guard withholding (next section).
   The other fold's samples are painted on the complete sites of that released panel. The result
   is a held-out carrier profile per sample and row (expected copies and carrier probability) that
   never uses the sample's own evidence. These calls serve only the profiles; they are not
   published.
2. **Cavity fills** (`assembly/completion/cavity_fill.py`). The block is cut into bins of
   `AssemblyConfig.completion.cavity.snps_per_bin = 50` sites. For each bin, every sample's
   pair-state weights come from the block's complete, informative sites outside the bin
   (leave-bin-out). At each site of the bin the unresolved alleles are enumerated jointly: an
   assignment a has likelihood prod_n sum_s w_ns E_n(dosage(s, a)) under independent
   Bernoulli(1/2) priors, so every allele is drawn once and shared by all samples. A fill is a
   candidate when its posterior is at least
   `AssemblyConfig.completion.cavity.minimum_posterior_probability = 0.99`, at least
   `AssemblyConfig.completion.cavity.minimum_supporters = 2` carriers support it among the
   focal-informative samples whose likelihood depends on the allele, the row differs from every
   other row at an observable site outside the bin, and the site is neither multimodal nor withheld
   by the shell guard.
3. **Boundary links** (`assembly/completion/links.py`). Rows with identical calls form classes. A class of one block
   and a class of the next are linked when, on the samples informative in both blocks, their
   held-out copy numbers agree (mean squared discrepancy of expected copies and of min(copies, 1)
   at most `AssemblyConfig.completion.links.maximum_mean_cost = 0.099`) as a reciprocal unique
   optimum by a margin, pooled and in each fold, with enough informative samples and effective
   copies on both sides.
4. **Occupancy gate** (`assembly/completion/occupancy.py`). A row's candidate fills are released only when the row is
   demonstrably occupied. In the abundant route, the squared carrier probabilities of the
   informative samples sum to at least
   `AssemblyConfig.completion.occupancy.minimum_abundant_sum_squared_probability = 8.0` and at
   least `AssemblyConfig.completion.occupancy.minimum_abundant_high_confidence_carriers = 5`
   samples carry the row with probability 0.8 or more; all its candidates are released. In the
   rare-continuity route, a link of single-row classes reaches a row of an adjacent block that at
   least two samples informative on both sides also carry (local probability at least 0.9,
   neighbour at least 0.8, product at least 0.72), the best such neighbour is unique, and a fill is
   released where one of those samples is focal-informative. A block releases nothing when it has
   one row, or when local selection left unexplained copies there: some sample with depth has an
   unknown-copy slot and the block's unknown-copy fraction exceeds
   `AssemblyConfig.completion.maximum_discovery_unknown_mass = 0.0`.

Released fills change only the published calls (`released`, with the fill posterior as ALT
probability). Every likelihood of the linker, the joins, founder refinement and painting reads the
pre-fill `inference` calls, so a fill never strengthens the evidence that produced it, and the count
rule's row geometry measures distances on the same pre-fill calls
([founder-count rule](#founder-count-rule)): a fill changes the published panel, never an assembly
decision.
Completion runs in every assembly of a contig, contexts included; its summary (candidate and
released fills per block, vetoed blocks, gate reasons, site-posterior counts) is stored with the
prepared panels and logged.

## Founder-allele site posterior

Completion's EM and cavity fills, and local selection's context refit, need the posterior of the m
unresolved founder alleles b in {0, 1}^m at one site, shared by every sample
(`assembly/founder_alleles/site_posterior.py`). Each informative sample contributes a positive
quadratic Q_n(b) (its pair weights times its genotype evidence), and p(b) is proportional to
exp(a . b) prod_n Q_n(b) with log prior odds a. For m at most
`assembly.founder_alleles.site_posterior.MAX_EXACT_BITS = 10` every configuration is enumerated.
Above that the uncertain-core approximation applies, with no cap on the founder count:

- the mode comes from loopy belief propagation (one factor per sample) followed by steepest ascent
  over single and pair flips, started from the thresholded beliefs and from their full allele
  complement; the better optimum is kept;
- when the two starts end in different optima within
  `assembly.founder_alleles.site_posterior.SITE_CERTAIN_NATS = 20.0` of each other, the posterior
  has two comparable basins that no single or pair flip connects (typical where most carriers look
  heterozygous, as at paralogous sites), and the site releases nothing (multimodal abstention);
- founders whose single flip and every pair flip cost more than 20 nats stay at the mode; the
  remaining core is enumerated exactly when it has at most ten founders, otherwise over the
  Hamming ball around its mode of the largest radius with at most
  `assembly.founder_alleles.site_posterior.SITE_BALL_CONFIGURATIONS = 1024` configurations, but of
  radius at least `assembly.founder_alleles.site_posterior.SITE_BALL_MINIMUM_RADIUS = 3`
  (4,526 configurations for a 30-founder core);
- in the ball, a release is withheld when the marginal lies within
  `assembly.founder_alleles.site_posterior.SITE_SHELL_GUARD = 3.0` times the weight of the
  outermost-shell configurations that flip the founder of the release threshold.

Exact marginals of this model are #P-hard in general, so above ten free founders the marginals are
exact only over the enumerated support. The work per site is O(N m^2) per propagation iteration and
ascent step, O(N (m^2 + 2^10)) for a core enumeration and O(N m^3) for a radius-3 ball. In
completion's EM the support of every site is fixed for the whole EM (chosen under the starting pair
posteriors), so the evidence lower bound stays a lower bound.

## Hierarchy

Level 1 starts from one single-block component per prepared panel, with every local row as a
founder. Each level (`assembly/hierarchy/`) then proceeds in three steps.

1. **Partition.** A sample is informative for a component when its evidence is non-uniform at one
   of the component's usable linking-proxy sites: discovery sites where some founder has a
   pre-fill call, at most `AssemblyConfig.linker.max_sites = 2000` per component (an even stride
   over the usable sites of a longer component). A boundary whose two sides share fewer than
   `AssemblyConfig.min_boundary_informative_samples = 4` informative samples becomes a phase break
   (`insufficient_joint_informative_samples`) and stays one at every later level. The components
   are cut into batches of `AssemblyConfig.batch_size = 10` adjacent components that never cross a
   break; a batch's samples are those informative in every one of its components, since inference
   spans the whole batch.
2. **Joins.** Every batch of at least two components with at least one sample is joined
   ([linker](#linker), [endpoint beam](#endpoint-beam), [certified panel search](#certified-panel-search)).
   A single component, or a batch without informative samples, passes through unchanged; its
   boundaries are not marked as breaks.
3. **Output.** A joined batch becomes one component whose rows compose the chosen paths with the
   input components' rows.

The loop stops after a level when one component remains, or when a level from the second on did
not reduce the component count. A context assembly stops at its level count. The final assembly
runs at least `AssemblyConfig.minimum_levels = 4` levels while they reduce the count, and beyond
them continues only while some boundary between adjacent components is not a phase break. With
batches of ten, four levels join up to 10^4 blocks into one component; a longer break-free segment
is joined at level 5 and above. In the final assembly, founder refinement runs after every level
and its output is the next level's input; the level at which the assembly stops is its last level,
the only one at which refinement may add founders.

A component boundary is either a data-unsupported break (the reason above) or carries no reason:
a chromosome end, or a boundary the hierarchy left unjoined for another reason (a batch without
informative samples, or a batch edge at the last level).

## Linker

The linker (`assembly/linker/`) fits dense haploid founder transitions between the blocks of one
batch (its input components, read through their linking proxies) from the batch's informative
samples.

- **Chains.** For every gap g = 1 .. G the blocks split into g residue-class chains (blocks i,
  i + g, i + 2g, ...). G is n - 1 at level 1 and the beam's maximum gap at later levels: only the
  gaps the beam reads are fitted. In a chain a sample's state is an ordered founder pair; within a
  block it evolves along the block's emission sites, and between consecutive blocks of the chain
  both homologues move independently through one haploid transition matrix T_i (K_i x K_{i+g}),
  so no K^4 diploid transition is formed. Every gap is its own model and EM.
- **Within a block.** Emission sites are the usable proxy sites. Between adjacent emission sites
  at most one homologue switches founder: with theta the interval's genetic distance (Morgans;
  max(1 bp, distance) times the fallback rate without a map), capped at 0.5, the stay and
  per-founder switch weights (1 - theta)^2 and theta (1 - theta)/(K - 1) are normalised to row
  mass 1. Theta is the distance of one meiosis; it is not multiplied by `AssemblyConfig.n_generations = 3`,
  which only the beam's gap and local selection's path model read. A two-state quality indicator
  follows a continuous-distance Markov process with stationary error fraction
  `AssemblyConfig.linker.error_fraction = 0.01` and mean error tract
  `AssemblyConfig.linker.error_tract_bp = 20000.0` bp: in the normal state the pair emits through
  its predictive class, in the error state the mean of the three genotype likelihoods (independent
  of the founders); ancestry transitions apply in both states. Each block starts the indicator at
  stationarity and sums it out at its far end. Inference is sum-product over both ancestry and
  quality; the backward scan is the adjoint of the forward operator.
- **Emissions.** The predictive likelihood of a pair class is mixed with a uniform genotype,
  (1 - r) P + r/3 with `AssemblyConfig.linker.robustness = 0.01`, without a further log floor. The
  seven classes are: both alleles called (dosage 0, 1 or 2); one called allele and one unknown
  (averaging dosages 0 and 1, or 1 and 2); two unknown alleles of different latent rows (weights
  1/4, 1/2, 1/4); two copies of one unknown latent allele, the same founder or two founders
  copying the same prepared row (weights 1/2, 0, 1/2). Unknown alleles are Bernoulli(1/2). This
  is a per-observation predictive approximation, not exact integration of shared latent alleles
  across samples; it changes linkage evidence and never imputes an allele.
- **EM.** Transitions start uniform and each chain starts from a uniform pair prior. The E-step
  runs forward and backward over every chain (the backward contraction uses the transpose of T,
  not a reverse conditional). The M-step takes the posterior homologue edge counts, adds the
  pseudocount `AssemblyConfig.linker.pseudo_count = 0.1`, normalises each row, floors it at log
  probability `AssemblyConfig.linker.min_log_probability = -10.0`, renormalises and mixes in a
  uniform row, (1 - u) p + u/K_{i+g} with `AssemblyConfig.linker.uniform_mix = 0.01`. The update is
  damped toward the previous transitions: T becomes (1 - eta) T + eta T_target with
  eta = max(0.9^t, 0.1) at iteration t = 0, 1, ... (`AssemblyConfig.linker.learning_rate_decay = 0.9`,
  `AssemblyConfig.linker.minimum_learning_rate = 0.1`), so the first update is undamped. The EM
  stops when no transition probability and no reverse summary (the column-normalised regularised
  edge evidence, read only by this stop rule) changes by more than
  `AssemblyConfig.linker.min_change = 0.001`, and after at most
  `AssemblyConfig.linker.max_iterations = 20` iterations.

The linker hands the beam the natural logs of the fitted probabilities, rounded to float32. Its
dense propagation and edge counts are cubic in the founder count (O(N (K_i^2 K_j + K_i K_j^2)) per
edge and iteration); scaled arithmetic with a log-domain fallback implements the same model.

## Endpoint beam

The endpoint beam (`assembly/join/beam.py`) enumerates candidate chromosome paths through a batch
(one founder per input component). A path scores the sum over blocks k >= 1 of the mean of the log
transitions into block k from the earlier blocks within the maximum gap. Pass 1 builds paths left
to right and keeps, at every block, at most q paths per endpoint founder
(`AssemblyConfig.join.paths_per_endpoint = 16`). Pass 2 refines backwards from the last block:
every kept path tries every founder at block k with the rest fixed, and the pooled candidates are
selected with the same endpoint quota. Every path selected at any step is archived, so full paths
through interior states survive; the archive, sorted by score, is the batch's candidate pool C0
(at most n times the per-block sum of endpoint quotas, linear in K). Candidates are ranked by
descending score with ties broken by the lexicographic path; scores accumulate in float32 in
past-block order. The panel search's tie rules downstream depend on this order.

At level 1 every gap enters a step. At later levels the maximum gap is
max(1, 1 + floor(t / r)), where r is the expected number of recombinations per step (the mean
genetic length of the batch's components times n_generations) and
`AssemblyConfig.recombination_tolerance = 0.5` is t; a batch whose components carry no
recombination uses every gap.

## Join objective

A panel P of K paths through the batch is scored per sample by a Viterbi programme over unordered
pairs of panel paths (`assembly/join/objective.py`):
V_s(P) = max over state sequences of [sum_t e_s(z_t, t) - penalty x (number of state changes)],
with a free start. The join log likelihood is LL(P) = sum_s V_s(P). The emissions are binned
(`assembly/join/emissions.py`): each input component's sites are cut into bins of
max(clamp(floor(mean component sites / 20), 10, 100), ceil(batch sites / 2000)) sites (at most about
`AssemblyConfig.join.max_bins = 2000` bins per batch), and the emission of a pair at a bin is the
sum over the bin's usable sites of max(log(0.99 P(g | pair) + 0.01/3), -2) - log(1/3), with the
linker's seven predictive classes on the pre-fill calls; uniform evidence contributes nothing. The
penalty is 20 nats while the batch spans at most 5,000 sites and 20 sqrt(sites / 2000) above. With
symmetric genotype emissions and a uniform penalty the ordered states (a, c) and (c, a) score
equally, so the K(K+1)/2 unordered pairs carry the programme exactly. Ties are broken
deterministically: the switch source is the first maximiser in pair order, and a state stays when
staying is at least as good as switching.

## Certified panel search

The panel search (`assembly/join/search.py`) chooses a batch's founder paths from the candidate
pool: same-count moves are judged by LL alone, the founder count by the
[founder-count rule](#founder-count-rule).

**Start.** For every anchor block j, taking the beam paths in beam order and keeping each one whose
row at block j is not yet covered gives an anchor panel; the distinct anchor panel of largest LL
(ties: the first anchor) starts the search. It only seeds the search: the count rule decides the
panel size.

**Same-count moves** keep the number of paths: replace (path f becomes pool path c), local (f takes
row r at block b), one-sided suffix copy (f takes g's rows from block t on), splice (f and g
exchange their rows from block t on) and suffix cycles. A move is admissible only when the panel
keeps K distinct paths and every changed path keeps a private cell (a row at some block that no
other path uses) and differs from every other path at more than the near-identity rate of called
sites (`AssemblyConfig.count_rule.near_identity = 0.01`); otherwise the move would create a path the
count rule deletes. It is accepted only when it raises LL beyond the score's float resolution,
2 dLL > max(1e-8, 1e-9 (1 + 2 |LL|)).

**Tier 1** is the bounded sweep. Nine proposal kinds are ranked by cheap statistics (conditional
fixed-painting gains, candidate novelty under a mate model, exact no-switch replacement gains,
boundary votes of an exploratory painting at penalty min(penalty, 10), greedy and cycle
permutations); of the top `AssemblyConfig.join.full_scores_per_kind = 16` proposals of each kind,
the distinct admissible ones are scored exactly, and the best improving one is accepted.

**Count step.** At every tier-1 stall the count step runs. Paths without sequence of their own (no
private cell, or within 1% of another path) are deleted one after another, and the death candidate
of largest V(P minus f) (ties: the lower path) is then decided by the rule. Only when no path died
is the pool birth of largest total gain among those the rule accepts added; births are bounded per
sample (two-region bounds) and scored exactly only where the bound leaves them open. A count change
returns the search to tier 1.

**Tier 2** runs when the count step is quiescent. The pool is extended by a beam re-run with quota
q times the number of panel paths sharing each endpoint (births by the added paths are decided
first), and every same-count move of an explicit neighbourhood (replacement by any pool path, any
one-block row change, suffix copies and exchanges at every boundary) is bounded through exclusion
forwards; moves the bound cannot exclude are scored exactly, best bound first, and the best
improving admissible move is accepted (back to tier 1).

**Duplicate-seed births.** At a panel certified for both, pool births without a private cell that
would pass on their total gain indicate a chimera: one panel path covers cells of an unrepresented
founder. Seeds are every panel path while K is at most
`AssemblyConfig.count_rule.ranked_refits = 9`, otherwise the main coverers of the strongest
indicators. Each seed is refitted from P plus a copy of the seed (certified best improvement over
the moves that change the seed or its copy, run concurrently), then settled: paths without
sequence of their own are deleted and the death candidate of largest V(P minus f) must survive its
death test. A settled panel is accepted when it survives, keeps at least K paths and raises LL.

**Pool saturation.** Before a panel is certified, the beam is re-run with doubled quotas (32, 64,
128 and 256 paths per endpoint, times the endpoint's multiplicity), and the paths each rung adds
are tested as replacements (by LL) and births (by the count rule). A move found at a rung raises the
quota to that rung and returns the search to tier 1. A join whose 256 rung accepted a move climbs
on to 512 and 1024 (`assembly.join.search.POOL_QUOTA_DOUBLINGS = 4`,
`assembly.join.search.POOL_EXTENDED_DOUBLINGS = 6`). The panel is certified when no rung up to its
ladder top adds a move.

**Backstops** are reported and never decide a normal result: at most
100 + 20 max_b K_b same-count sweeps (tier-1 sweeps and accepted tier-2 moves) since the last count
change or saturation move (`assembly.join.search.SWEEP_BACKSTOP_BASE = 100`,
`assembly.join.search.SWEEP_BACKSTOP_PER_ROW = 20`), and a move accepted at quota 1024 (the pool
backstop). Each join records the flags `certified`, `backstop_reached`, `ladder_extended` and
`pool_backstop` with its tier-2 passes, survivors, exact scores, pool sizes and every count decision
(margins on the total and sequence parts, band, ambiguity); each level logs two census lines.

Certification is relative to this neighbourhood and to the beam's pool, not a global optimum:
paths the beam never proposes, three-way boundary cycles and multi-founder births lie outside it
(tier 1 still proposes cycles, and duplicate seeds split chimeras). A founder that is a pure block
mosaic of other founders within one batch is not carried by that batch; later levels and founder
refinement can re-split it.

## Founder-count rule

The join and founder refinement decide founder counts with one rule (`assembly/count_rule/`). A
decision compares a larger panel X+ with a smaller X- that lacks one founder f. With per-sample
scores (the join's V_s, refinement's F_s), the total gain is d_s = score_s(X+) - score_s(X-) and its
sequence part is q_s = score_s(X+) - score_s(X+^sigma). The shadow X+^sigma replaces every cell
(block, row) of X+ that a reference panel does not use by the reference's row at that block nearest
in Hamming distance over sites called in both (ties: the lowest row); the reference is X- in the
join, and in refinement X+ without f for a deletion or the incumbent for an addition. A founder
without a private cell is a pure mosaic of the others and has q = 0 exactly, so switch savings
alone never justify a founder.

**Effective-carrier test.** The per-founder cost is c = s x (mean sites per block / 200) x S x B
for a span of B blocks and S samples, i.e. c = s S L / 200 for a span of L sites, with
`AssemblyConfig.cost_scale = 0.5` as s. In the join the blocks are the batch's input components and
S counts its informative samples; in founder refinement the blocks are the component's prepared
panels and S counts the component's informative samples (samples with evidence at one of its complete
sites). With n_eff = (sum_s d_s+)^2 / sum_s (d_s+)^2, the
participation ratio of the positive differences, a difference vector passes when
m = sum_s d_s / c - n_eff / (2S) > 0 and m' > 0, where m' is the same margin with the sample of
largest d_s left out (S unchanged). The cost is thus charged only to a founder's effective carriers
(c / (2S) per carrier, L / 800 nats at the default scale; with n_eff = S the margin equals BIC's
sum d / c - 1/2), and no single sample can decide a founder. X+ is kept (deletion rejected) or
accepted (addition) only when the test passes on d and on q.

**Ambiguity.** Samples are resampled with replacement
(`AssemblyConfig.count_rule.bootstrap_resamples = 2000`, each decision with a deterministic stream
from a SHA-256 of its key) and min(m_d, m_q) is recomputed; a call whose percentile interval
(`AssemblyConfig.count_rule.bootstrap_level = 0.95`) contains zero in its interior is ambiguous.
The point decision stands.

**Near identity (join only).** A path within 1% called-site mismatch of another panel path over the
batch span owns no sequence (q = 0): it is never added, it is deleted first, and no same-count move
may produce one. Founder refinement does not use this shortcut (it scores the shadow exactly).

**Count cycles.** Deleting a path that an addition of the same search created closes a count cycle:
the rule, called against different panels, has both added and deleted that row. The deletion
stands; the join then adds no further births, and refinement ends its additions. A cycle is
undecided when its closing deletion is ambiguous and it began with a decided deletion (not with an
addition, and not with the deletion of a path without sequence of its own, which is not a
decision); undecided cycles are counted in the census and published in `founder_counts.tsv`.

The row geometry (nearest rows, shadows, the near-identity and novelty tests, same-count admissibility) measures
distances on the pre-fill inference calls, the calls every likelihood reads: a completion fill is a published call
no likelihood sees, so it never makes two rows differ or agree.

## Join records and reuse

Every joined batch is recorded as `joins/<digest>` at the contig root as soon as it completes. The
digest is a SHA-256 of the batch's explicit typed inputs: the join and linker settings, count rule,
cost scale, gap settings (levels 2 and above) and genetic map identity, the code identity of the
assembly package, the evidence record's identity, the batch's informative samples, the content
digests of its prepared panels (site range, positions, discovery sites, released and inference
calls) and the input components' relative ranges and rows. Joins are deterministic, so equal digests
give equal joins. The digest does not depend on where the batch lies in the assembly, so an
identical batch of another assembly of the same contig (a context assembly and the final one) reuses
the record instead of recomputing it.

## Records

Every assembly binds its identity before writing: the stage configuration (without the refinement
settings for a context assembly, which never refines), the role and level count, the upstream
identities (the evidence record, the local panels: the `local_selection/l2` record for the final
assembly and a content digest for contexts, and the genetic map) and the code identity of the
assembly package (with `model/` and `maths/`). A different bound identity stops the run instead of
reusing records. Records under the section prefix P (`assembly`, or the context's prefix):

| Key | Content |
|---|---|
| `identity/P` | the assembly's identity |
| `P/completion` | prepared panels and the completion summary |
| `P/l<k>/plan` | level k's batches: (first input component, end, join digest or none) |
| `joins/<digest>` | one joined batch, shared by the contig's assemblies: rows over its prepared panels, panel log likelihood, search summary |
| `P/l<k>/join` | level k's hierarchy output: components and the level record (plan, join summaries, log likelihoods, reused joins, informative samples per boundary, stop reason) |
| `P/l<k>/refine/...` | founder refinement's records of level k ([founder refinement](founder_refinement.md#records-and-census)) |
| `P/l<k>/refined` | level k after founder refinement (final role): components and refinement summary |
| `P/final` | the stage product (final role): the last level's refined components with count calls, and the level records |

Each record is written as soon as its step completes, so an interrupted assembly resumes from its
last recorded step: completion, then each level's output, joined batch and refined level. A level
whose records are present is decoded, not recomputed.

## Implementation and cost

| Module | Role |
|---|---|
| `assembly/stage.py` | `assemble(inputs, config, role=..., levels=..., records=..., budget=...)`: roles, level loop, stop rule, records |
| `assembly/completion/` | cross-fit, cavity fills, boundary links, occupancy gate |
| `assembly/founder_alleles/` | the shared site posterior and its dosage polynomials |
| `assembly/hierarchy/` | partition, batch tasks and digests, level execution |
| `assembly/linker/` | emissions, within-block scans, cubic transition kernels, EM |
| `assembly/join/` | beam, binned emissions, objective, certification bounds, moves, proposals, count step, search |
| `assembly/count_rule/` | the shared count rule: criterion, bound tests, geometry |
| `assembly/founder_refinement/` | [founder refinement](founder_refinement.md) |
| `model/components.py` | `PreparedPanel`, `FounderComponent`, `ComponentPanel`, `CountCall`, `materialise` |

**Parallel structure.** Completion runs its blocks and then its boundary links in a pool of
single-threaded worker processes. A level's batches run in a pool of worker processes that share
the assembly's CPUs dynamically (a single pending batch runs in the coordinating process with the
whole budget); with `[execution] batch_queue` they may run on other nodes. Inside a join, the linker's block scans, gap fits and forward/backward recursions and the
search's duplicate-seed refits run concurrently on shares of the live thread budget. Completion
tasks and batches are admitted by their estimated memory, so a run slows down rather than
overcommitting memory; this never changes a result.

**Memory.** The contig's assembly evidence (`assembly/evidence.py`, `AssemblyEvidence`: the neutral float64
likelihoods, their float32 copy and the observation mask) lives in shared memory from the start; completion,
hierarchy and founder-refinement workers attach it instead of copying it, and a completion task copies only its
blocks' columns. Every task reserves an upper estimate of its peak memory with the run's memory governor before it
starts:

- a completion block (`assembly/completion/stage.py`, `block_memory_bytes`): its evidence columns, the cavity
  fill's (N, K, K, bin sites) emissions with their temporaries, the per-bin pair-state scores and the cross-fit's
  pair-state arrays; a boundary link (`link_memory_bytes`); and the coordinator's resident arrays;
- a joined batch (`assembly/hierarchy/level.py`, `batch_memory_bytes`): the float32 evidence of its informative
  samples over its sites, the linker's emissions and gap-fit messages, the search's binned emissions
  (S, K_b, K_b, m_b), painting and exclusion arrays, painting back pointers bounded by
  `AssemblyConfig.join.tensor_budget_mb = 256`, and the move arrays and beam lists up to the saturation ladder's
  normal top (quota 16 x 2^4). A batch's reservation is an envelope: the search's nested reservations draw on it
  first, and a search whose ladder climbs past the normal top reserves the extra quota itself
  (`assembly/join/search.py`, `ladder_extension_bytes`).

When memory is tight, tasks wait for their reservations; results never depend on that pacing.

**Complexity.** N is the number of samples (a join reads its batch's informative samples), L the
number of sites, K the largest working panel of a step (including temporary panels; not a
biological founder count), B the components per batch (at most 10), m the join bins of a batch
(about 2,000 at most), q the endpoint quota, |C| the candidate pool (O(q B K) paths) and I the linker
iterations (at most 20). Nothing is exponential in K: the only exponential term is the site
posterior's enumeration, capped at 2^10 configurations.

| Step | Work | Largest power of K |
|---|---|---|
| Completion EM, per block and iteration | O(N L K^2 + sum_l C_l), C_l at most 2^10 configurations or the uncertain-core support | K^2; O(N U^3) per site with U > 10 unresolved alleles |
| Cavity fills, per block | O(N K^2) pair emissions per site; O(N K^2 + N 2^U) per site with U <= 10 unresolved alleles (exact carrier sensitivity O(U N 2^U)), O(N U^3) above | K^3 |
| Linker, per gap edge and iteration | O(N K^3) propagation and edge counts; within-block scans O(N L_b K^2) | K^3 |
| Endpoint beam | O(n q K^2 G^2) path scores plus sorting (G the largest gap) | K^2 log K |
| Tier-1 sweep | O(N m K^2 + K^2 B log K) | K^2 |
| Tier-2 pass | O(N m K (\|C\| + K^2)) | K^3 |
| Count step | O(N m K \|C\|) bounds plus exact scores where a bound is open | K^2 for fixed q and B |

Founder refinement's costs are listed on its own page ([founder refinement](founder_refinement.md#implementation-and-cost)).
Search budgets that do not grow with K (16 exact scores per proposal kind per sweep, 9 duplicate
seeds) change only the path to a certified panel: tier 2 bounds the whole neighbourhood.

**Memory.** The evidence view holds 37 N L bytes (float64 and float32 likelihoods and the observed
mask), shared by every worker. A join's binned emissions hold 8 S sum_j K_j^2 m_j bytes (for example
4.6 GB at S = 320, K = 30 and 2,000 bins), its exclusion forwards 8 S m bytes per excluded path (at
most K + 1 paths), its painting back pointers at most `AssemblyConfig.join.tensor_budget_mb = 256`
MiB per sample chunk, and the pool-saturation step O(B q K^2) move and beam entries at its top rung.
The linker keeps seven log emissions and seven weights per sample and proxy site.

**Determinism.** Per-sample values are computed by one thread each and summed in sample order;
concurrent refits are evaluated in seed order; caches never change values; bootstrap streams are
seeded from the decision keys. Results therefore do not depend on thread counts, worker counts,
completion order or memory pacing.

## Interpretation and limitations

- A phase component is a stretch over which the data link founder labels; labels are row indices
  local to the component and say nothing about labels in other components. A break with the reason
  `insufficient_joint_informative_samples` is a data limitation (too few samples informative on
  both sides), not a biological event; a boundary without a reason is a limit of the hierarchy (a
  batch without informative samples, or a batch edge at the last level), not evidence for a
  break.
- K per component is the count the count rule supports on these samples. It is not an estimate of
  the number of biological founders: absent local alleles, unsampled ancestry, rare founders carried
  by few samples, founders missing from the local panels and long-range mislinking all affect it.
  An ambiguous count call (band containing zero, count cycle) marks a count the data cannot
  separate from its neighbour.
- The linker's emissions integrate unknown alleles per observation, so uncertain paths can be
  preferred in ambiguous regions; the linker is not a calibrated model of path probabilities, and
  the beam and panel selection maximise. Correlated errors that conceal a genuine crossover remain
  an identifiability limitation.
- Certified panels are certified relative to the explicit neighbourhood and the beam's pool; a
  higher likelihood or more called alleles do not by themselves establish a better biological
  reconstruction.
- Completion fills are published calls, not observations: they never enter a likelihood. A
  multimodal site releases nothing even where the founder alleles are identifiable by other means.
