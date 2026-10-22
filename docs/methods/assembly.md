# Assembly

Assembly turns the path-selected local founder panels of one contig (one small panel per discovery
block, each with K_b local founder rows) into chromosome-scale founder haplotypes, split only into
phase components whose boundaries the data cannot bridge. It prepares local panels and joins them hierarchically. Founder
completion fixes the prepared panels: the local calls plus the unknown alleles the data determine
well enough to publish. A hierarchy of levels then joins batches of adjacent components: a
distance-aware linker fits founder transitions between the blocks of a batch, an endpoint beam
enumerates candidate chromosome paths, and a certified panel search chooses the batch's founder
paths, with their number decided by the pipeline's founder-count rule. In the contig's final
assembly, joined components pass directly to the next level and final output. Founder-path
refinement is disabled in 2.7, with its source retained for the planned 2.8 rebuild. The same computation has two roles: the final assembly (`Role.FINAL`), the stage
product that [painting](painting.md) reads, and the context assemblies of local selection's
feedback rounds (`Role.CONTEXT`, through level 1 or 2), whose components
seed the next round of [local selection](local_selection.md). No pedigree, cohort label,
simulation truth or target founder count enters.

## Inputs and outputs

| | Final assembly | Context assembly |
|---|---|---|
| Local panels | the `local_selection/l2` panels (`LocalPanel`, one per informative block) | the panels of the calling round |
| Levels | until the stop rule ends the loop ([hierarchy](#hierarchy)) | 1 (round `l1`) or 2 (round `l2`), `LocalSelectionConfig.context_levels = (1, 2)` |
| Records | under `assembly/` | under `local_selection/l1/assembly` and `local_selection/l2/assembly` |

Every assembly also reads the contig's evidence view and genetic map. The evidence view
(`assembly/evidence.py`, built once per contig from the evidence record's founder view) holds the
normalised genotype likelihoods where a sample is observed at a discovery site and exactly 1/3 at
every other cell (unobserved, not a discovery site, or a sample the contamination QC excluded from
founder construction). Completion and the batch partition read it in float64; the linker and the joins read one float32 copy of the same values. The view also retains compact
per-sample keys of integer total read depths on founder-visible discovery sites for completion's
cross-fit partition. Counts are decoded transiently to construct those keys; the full count array
is not retained by assembly. The genetic map gives the
linker's within-block switch probabilities and the beam's maximum gap: distances come from the map
file where one covers the contig (with the fallback rate `[genetic_map] rate_cm_per_mb` beyond its
ends) and from the fallback rate alone otherwise.

The product of the final assembly is the record `assembly/final`: the phase components
(`FounderComponent`, with their count calls) over the prepared panels of `assembly/completion`.
Other stages and the evaluator read it through `assembly/records.py`: `load_product`,
`founder_panels(file, level)` (level k: the level-k joined product; no level: the final product), `level_records` and
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
latest join: K before and after, explicit deletion comparisons and accepted pool births (kind,
K, kept or added, margin, bootstrap band, ambiguity), count cycles and overall ambiguity.
Seed and bound work remains in the full join census. A component that never joined has no
count call; its row count is the local panel's count, with no new confidence claim.
`founder_counts.tsv` names this source (`join` or `local_panel`). Its `CountStepReason` is
`join_search_unverified` for unfinished search/verification; `FounderCountAmbiguous` is empty
when such work is unverified without demonstrated ambiguity. Evaluated different-K alternatives
remain ambiguous. The latest join's undecided cycles are already in `JoinUndecidedCycles` and
are not counted again in `UndecidedCountCycles`. Stage inputs are never modified: completion returns prepared
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
   folds (`AssemblyConfig.completion.crossfit.fold_seed = 1729`). The primary key is a seeded
   SHA-256 of the observation pattern on the contig; ties are refined by a seeded key of the exact
   integer total-depth vector (REF + ALT) on founder-visible discovery sites, zero elsewhere.
   Samples with identical observation and depth content stay in one group. In content-key order,
   each whole group goes to the fold with fewer informative members so far; an equal load selects
   label 0. For balancing, informative means at least one founder-visible observed site,
   independent of likelihood values. Samples with no founder-visible observation have zero balancing weight and fixed
   label 0. With distinct informative keys this gives an alternating balanced split. Residual
   groups are never divided by input position, even if their genotype likelihoods differ.
   Sample names, input order, inferred genotypes and floating-point likelihood values do not
   enter the split. It is invariant to sample permutation, renaming and convention reversal;
   GL-only noise cannot change the integer depth keys. Array-only callers without counts hold
   whole observation-pattern groups together and record that key source explicitly.
   On one fold a variational EM over
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
   never uses the sample's own evidence to learn its training panel. When the training complement
   has no informative observations, it cannot learn missing alleles: the original pre-fill panel
   is used unchanged for held-out painting and absent informative training is flagged. This
   fallback changes neither the separate cavity-fill model nor the occupancy criteria. Fold
   group sizes, informative loads, key source and absent-training diagnostics accompany the
   completion summary. These learned calls serve only the profiles; they are not published.
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
probability). Every likelihood of the linker, the joins and painting reads the
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

Completion retains this support as a centre, an ordered core and sparse flip sets. No growing
ball is expanded to a configuration-by-founder matrix. Quadratic scores use the value and local
fields at the centre plus the couplings among the flipped bits. Pair-dosage probabilities use
single- and joint-flip masses. Their work is O(K^2 + C r^2) per site and update, where C is the
support size and r its radius. Allele and shell marginals cost O(C r + K^2); cavity log odds use
positive interval sums of the unchanged support, avoiding subtraction of nearly equal masses.
Carrier sensitivity still evaluates the mode and its single core flips, O(N U^2).

The exact cube at U <= 10 remains bounded by 1,024 assignments. Larger-radius balls also remain
inside the existing 1,024-configuration envelope; growing supports have radius three and
C = 1 + U + choose(U,2) + choose(U,3). Thus completion has at most cubic dependence on K,
with fixed sample count, block length and existing iteration limits. This changes evaluation
order only: support, priors, thresholds, convergence rules and released scientific decisions
are preserved. Continuous agreement is scientific, not bitwise.

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
is joined at level 5 and above. In both roles, each level's joined components and release
alternatives are the next level's inputs. The final product is the last joined output.
There is no later founder-path refit or extra final founder-addition pass.

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
  mass 1. Theta is the distance of one meiosis; it is not multiplied by `AssemblyConfig.gap_horizon_scale = 3`,
  which controls only the linker and beam's inter-component gap horizon. A two-state quality indicator
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
  stops when the undamped fixed-point residual `max(abs(M(T) - T))` at the returned forward
  transitions is at most `AssemblyConfig.linker.min_change = 0.001`, with M the regularised
  forward M-step, or after `AssemblyConfig.linker.max_iterations = 50` applied updates. The
  cap increases from 20 to 50 in 2.7; tolerance and damping are unchanged. The higher cap
  allows further fitting but does not guarantee convergence. Column-normalised reverse summaries no
  longer enter the stopping rule: they are not parameters of this fixed-point problem. A uniform
  starting point that already satisfies the criterion takes zero updates.

Each fitted gap persists a `linker_em` event in its join search summary (or the retained-input
shadow record when the beam cannot establish a joined panel). It records the number of
updates, whether the iteration cap was reached, whether the fit exhausted the cap without satisfying
the stopping rule, the last damped **forward** change (absent for zero updates), and the undamped
fixed-point residual (also per edge). Each iterate's target is computed once and reused for its
update; there are updates + 1 E/M evaluations, including the final stopping evaluation. `converged`
means the returned residual is within tolerance; reaching the cap alone does not establish this.
All incoming parent branches retain these records, including branches that were not selected.
Existing checkpoints without these events have unmeasured convergence diagnostics; zero must not
be inferred from their absence. The changed stopping rule can change beam candidates and scientific
outputs, so it requires validation with the integrated candidate. The optional
`assembly.linker_em_stop` census now records residual-minus-tolerance distances, matching the
reported stopping rule rather than the previous damped-change distances.
Completion retains its own cross-fit diagnostics; these linker records describe boundary fits.

The linker hands the beam the natural logs of the fitted probabilities in float64. Its
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
(bounded complete families in 2.7; see below). Candidates are ranked by
anchored score tie classes with content ordering inside each class; scores accumulate in float64 in
past-block order. The panel search's tie rules downstream depend on this order.

Release2.7 represents constant-transition Cartesian families without enumeration and stops a
complete oversized frontier without first-N deletion. A current incumbent is retained and the
family is released as unresolved continuation. An initial nonconstant forward overflow retains
the input components unjoined because no joined incumbent yet exists. The binding policy is
output-changing; [assembly work and memory](assembly_performance.md) describes its bound and gates.

At level 1 every gap enters a step. At later levels the maximum gap is
max(1, 1 + floor(t / s)), where s is the mean genetic span of the batch's components in
Morgans times `AssemblyConfig.gap_horizon_scale = 3`, and
`AssemblyConfig.recombination_tolerance = 0.5` is the dimensionless budget t. The scale has
units of inverse Morgans and controls the proposal horizon: increasing it shortens the gaps
considered by both linker fitting and the beam. A zero scaled span uses every gap; if no
component spans two sites, the maximum gap is 1. This is a search rule, not a sample's meiotic
depth or a transition likelihood. It is independent of local selection's
`LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0`. Neither setting is derived from
generation labels or estimated from painting switches.

Both parameters are internal constants, not public TOML settings. The retained defaults are
copying intensity 3.0 and gap scale 3. A bounded chr19 study on std, F3 and TroMau finds no
principled gain from gap scales 1 or 6: both retain std accuracy and TroMau G0 pseudo-truth
consistency, while scale 1 trades better F3 genotype recovery for worse founder recovery.
These are painting-stage comparisons, not final-export accuracy gates or proof of an optimal scale.
Changing the gap scale bounds proposal search; it does not change the copying transition model.

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
a later join can reconsider it.

## Equivalent seed histories (2.7)

The TroMau chr7 L2 context join exposes a result-selecting convention in 2.6. Two
accepted duplicate-seed refits reach exactly the same canonical settled panel, but
only one encounters an unresolved two-panel tier-2 family. The global reversal hook
changes which equivalent seed witness is selected, so promoting only that witness's
histories changes `release_alternatives` from zero families to one. A bounded replay
of saved 2.6 inputs reproduces this while keeping all nominal rows and the join
likelihood identical. This is separate from the
[local-selection convention checks](local_selection.md#observed-effects-of-the-start-cap-and-inner-winner-order).

For a selected seed endpoint, the join now retains the union of unresolved source
families from every accepted seed candidate with that exact canonical settled panel.
Equality of likelihood alone does not qualify. Rejected seeds and different endpoints
do not contribute. This removes the witness-order convention without adding a search,
changing a count test, or declaring unexplored continuations equivalent. The union
uses only the candidates and family records already evaluated by the count step.

The context-refit consumer still uses the representative components and prepared
panels, not the alternatives sidecar. The observed context case therefore has no
released-local-call change. Final founder release does read ambiguity metadata;
its existing distinction between evaluated terminal disagreement and unperformed
continuations remains in force. This fix does not claim a downstream accuracy gain
or complete coverage of every remaining assembly convention.

## Live alternatives and founder release (2.3)

Proposal budgets now retain complete `join_proxy` classes. A cut class that would exceed
twice the quota is dropped whole unless it is the leading class; an oversized leading
class is kept whole. Painting-derived candidates are merged across both traceback
conventions before this cut. Admissibility is tested on each edit witness before target
panels are deduplicated. Exact scoring identifies a one-path replacement by the target's
path multiset, so an alternative edit description cannot choose a different scoring route.

A symmetric data decision records its complete discovered panel family. Up to four
complete search histories, including the primary history, are evaluated by replaying the
original inputs with the chosen decision prefix. Each replay preserves the ordinary
count-cycle and born-path rules. Immutable emissions are shared; mutable search state is
new. At a given supported K, final outputs are compared using `join_ll` classes on one
common scale. Different supported K values remain `ambiguous_k`; raw likelihood does
not choose the count. The representative uses smaller K, then the exact minimum number
of sample state changes among optimal paths. Canonical order names a representative only
after those priors, while every remaining alternative stays recorded.

At the next hierarchy level, all child choices are combined jointly. Two two-way children
require all four configurations. The higher-level allowance is sixteen complete incoming/search
leaves per component; it is never sixteen independent
four-history searches. Each incoming join configuration receives one search execution;
unperformed inner histories stay flagged. A product larger than sixteen is retained as an explicit
unresolved domain; no convenient subset is declared decisive. Informative masks are
computed on the usable-site intersection across every alternative before partitioning.
Samples must also satisfy the original proxy eligibility under every input alternative;
changing the proxy stride on the shared sites cannot introduce samples excluded by an original alternative.
Every compared join uses that common sample cohort, usable-site mask and prepared span;
excluded sites are also absent from its count geometry. Positions and bin boundaries stay fixed. Each configuration refits its linker.
Parent outputs are compared by their common likelihood within K; scores of separately
fitted linker models are not compared, and child likelihoods are not added to parent
likelihoods. Lost informative samples and usable sites, branch counts and costs are recorded.
Both FINAL and CONTEXT assemblies carry the surviving join alternatives directly.
Local selection currently consumes a context's declared representative.

This is a bounded comparison of discovered alternatives, not an enumeration of all global
optima. Uncontinued inner-seed histories, branch-budget overflow and any remaining untested
continuations stay explicit. At a count-quiescent terminal tie, the stated exact prior selects a discovered
neighbor and ordinary search continues with the existing count, cycle, born-path, quota and
sweep state. Flat transitions do not count as improvements or reset the search budget.
Already visited flat states cannot cycle; newly discovered neighbors may extend the live
family within the same four-history budget. A continuation that reaches a backstop remains
uncertified, and its source family is not consumed as a validated terminal alternative.

Founder export reads the final sidecar. A founder column and track extend only over
segments where represented terminal panels agree up to a whole-row permutation, preserving
duplicate multiplicity. Forward and reverse maximal common-interval scans supply a common
set of cuts and an uncertainty interval bracketed by observed SNP positions. Different
local content is withheld as unknown for evaluated final alternatives with established
final correspondence. For equal-K panels, unique exact row matches on invariant path
anchors preserve every allele cell that agrees across the complete evaluated family;
only differing or missing cells are withheld. Matching uses no approximate or truth-based
alignment. For unequal-K terminal panels, a displayed full-span path occurring exactly
once in every evaluated member retains its known prepared content, even if other paths
change that row ID's local multiplicity. Duplicate full paths and clipped fragments do
not establish this correspondence. This protection applies to each family before masks
are combined; prepared missingness, carrier masks and other families' withholding remain
in force. All continuity cuts and uncertainty intervals are retained. Without this exact
correspondence, varying prepared-row multiplicity retains the conservative local-row
mask in every representative. Withheld calls remain unknown during full-length r5 filling.
A source history whose continuation was not performed retains its incumbent calls and is
flagged `continuation_unverified`, with the evaluated source panels, their actual local
disagreement intervals, stated priors, cause and scope. The flag alone supplies neither a cut
nor withholding. For an evaluated biological family stopped by a joint/search budget,
the smaller-K and exact minimum-switch priors first identify the complete class in which
the representative can vary. A unique prior winner retains its calls. Within a remaining
tie, immutable prepared-row IDs and exact agreeing source-path anchors identify observed
local allele disagreement; only those differing cells are withheld. Unknown downstream
continuation and unmapped local IDs remain explicitly unverified. Continuity cuts require
an exact whole-row correspondence to a represented source panel. Computational count
overflow never invokes this rule. Source paths are not claimed to be final alternatives.
The manifest records
source inference component, segment index, source decision, uncertainty interval and newly
withheld cells separately from unobserved-carrier withholding.

Released segment identities are local. Downstream pedigree, offspring phase and recombination
inference remain conditional on the chosen assembly representative; this export operation does
not marginalize those analyses over founder alternatives. A 2.2 record without the sidecar is
reported as having unknown ambiguity coverage and must be reassembled to obtain this treatment.

## Founder-count rule

Each join decides founder counts with the rule in `assembly/count_rule/`. A
decision compares a larger panel X+ with a smaller X- that lacks one founder f. With per-sample
scores V_s, the total gain is d_s = score_s(X+) - score_s(X-) and its
sequence part is q_s = score_s(X+) - score_s(X+^sigma). The shadow X+^sigma replaces every cell
(block, row) of X+ that a reference panel does not use by the reference's row at that block nearest
in Hamming distance over sites called in both; the reference is X-. A founder
without a private cell is a pure mosaic of the others and has q = 0 exactly, so switch savings
alone never justify a founder.

**Effective-carrier test.** The per-founder cost is c = s x (mean sites per block / 200) x S x B
for a span of B blocks and S samples, i.e. c = s S L / 200 for a span of L sites, with
`AssemblyConfig.cost_scale = 0.5` as s. In the join the blocks are the batch's input components and
S counts its informative samples. With n_eff = (sum_s d_s+)^2 / sum_s (d_s+)^2, the
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

**Near-zero margins (2.7 decision 3).** There is no additional heuristic dead band around
zero. The effective-carrier point test remains `m > 0` and `m' > 0` for both total and
sequence gain. Count-margin candidate comparisons use the existing numerical tie quantum
(`count_margin`, 2^-26 in units of c); that ranking quantum is not a replacement threshold
for the point test. A numerical tie must be treated as its complete candidate family or
reported as unresolved, never settled by an arbitrary candidate order. Differences above
the applicable quantum are evidence, however small their absolute value. Bootstrap
uncertainty and unperformed comparisons remain separately reported. A new dead band would
change the model and requires independently demonstrated harm against simulation truth.

**Near identity (join only).** A path within 1% called-site mismatch of another panel path over the
batch span owns no sequence (q = 0): it is never added, it is deleted first, and no same-count move
may produce one.

**Count cycles.** Deleting a path that an addition of the same search created closes a count cycle:
the rule, called against different panels, has both added and deleted that row. The deletion
stands; the join then adds no further births. A cycle is
undecided when its closing deletion is ambiguous and it began with a decided deletion (not with an
addition, and not with the deletion of a path without sequence of its own, which is not a
decision); undecided cycles are counted in the census and published in `founder_counts.tsv`.

The row geometry (nearest rows, shadows, the near-identity and novelty tests, same-count admissibility) measures
distances on the pre-fill inference calls, the calls every likelihood reads: a completion fill is a published call
no likelihood sees, so it never makes two rows differ or agree.

For join count decisions in 2.3, every equally nearest row at every changed cell
belongs to the sequence-shadow domain. The complete Cartesian set is evaluated when it
fits `CountRuleConfig.shadow_combinations` (currently 2048). Agreement of all count verdicts
resolves the shadow; conflicting evaluated verdicts retain count uncertainty and use the
smaller-K representative. If the complete domain exceeds that computational budget, only
the incumbent's own shadow verdict is evaluated and retained; no order-selected prefix is
scored. The sidecar records `count_unverified`, cause, domain size, evaluated count and scope.
Computational incompleteness cannot itself delete a row or withhold a founder call. Failure
of the total-gain test already rejects the larger count regardless of the sequence shadow.
These are sequence-shadow evaluations, separate from the shared sixteen-leaf higher-level
budget. Bounds and sparse replacement scores retain their existing scientific meaning.

**Zero-sequence screening (R0-P03).** No additional priority rule is introduced for
private-cell founders with a diagnostic sequence-test failure. The existing pure-mosaic
case has exactly zero sequence gain by construction; a failed effective-carrier test
can instead reflect small positive, concentrated, negative or mixed contributions.
Saved hidden failures do not establish an identically zero sequence vector over every
nearest-shadow alternative. Prioritising them would change count selection, so it needs
that evidence and independent simulation validation before adoption.

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

Every assembly binds its identity before writing: the stage configuration, the role and level count, the upstream
identities (the evidence record, the local panels: the `local_selection/l2` record for the final
assembly and a content digest for contexts, and the genetic map) and the code identity of the
assembly package (with `model/` and `maths/`). A different bound identity stops the run instead of
reusing records. Assembly schema 3 identifies the pipeline without refinement. Run preflight
rejects older assembly identities before completed-stage shortcuts, including context identities;
use a new output/checkpoint root for this pipeline. Records under the section prefix P (`assembly`, or the context's prefix):

| Key | Content |
|---|---|
| `identity/P` | the assembly's identity |
| `P/completion` | prepared panels and the completion summary |
| `P/l<k>/plan` | level k's batches: (first input component, end, join digest or none) |
| `joins/<digest>` | one joined batch, shared by the contig's assemblies: rows over its prepared panels, panel log likelihood, search summary |
| `P/l<k>/join` | level k's hierarchy output: components and the level record (plan, join summaries, log likelihoods, reused joins, informative samples per boundary, stop reason) |
| `P/final` | the stage product (final role): the last level's joined components with their latest join count calls, and the level records |

Each record is written as soon as its step completes, so an interrupted assembly resumes from its
last recorded step: completion, then each level's output and joined batches. A level
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
| `assembly/founder_refinement/` | disabled source retained for the 2.8 rebuild; no pipeline or warm-up caller |
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
hierarchy workers attach it instead of copying it, and a completion task copies only its
blocks' columns. Constructing completion depth keys transiently decodes the int32 REF/ALT counts
(8 bytes per sample/site), uses one int64 depth row plus a mask as scratch, and retains only
32 key bytes per sample. Reconstruction planning and memory admission include that transient
count/scratch footprint. Every task reserves an upper estimate of its peak memory with the run's
memory governor before it starts:

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
E/M evaluations (at most 51: 50 updates plus the returned-iterate check). The table gives
leading kernels, not a blanket bound on the whole
search. Complete beam families have an explicit materialization bound in 2.7; the bin
dependence of exact two-row certification needs the qualifications below.

| Step | Work | Largest power of K |
|---|---|---|
| Completion EM, per block and iteration | O(N L K^2 + sum_l(K^2 + C_l r_l^2)); support construction O(N U^3) per growing-ball site | K^3; cubes and larger-radius balls have at most 1,024 assignments |
| Cavity fills, per block | O(N L K^2) bin statistics; per site O(N 2^U K^2) in the worst-case cancellation fallback for U <= 10, O(N K^3) above; compact priors and marginals O(C r + K^2) | K^3; the small exact cube is constant-bounded |
| Completion boundary links and occupancy | O((N + L) K^2 + K^2 log K) per boundary; O(N K L) occupancy per block with at most two atomic neighbours per row | At most K^3 |
| Linker, per gap edge and iteration | O(N K^3) propagation and edge counts; within-block scans O(N L_b K^2) | K^3 |
| Endpoint beam | O(n q K^2 G^2) path scores plus sorting (G the largest gap) | K^2 log K |
| Tier-1 sweep | O(N m K^2 + K^2 B log K) | K^2 |
| Tier-2 pass | replacement bounds plus adaptive exact structural screening; see qualification below | structural screening O(N m^2 K^3), fixed m |
| Count step | O(N m K \|C\|) bounds plus exact scores where a bound is open | K^2 for fixed q and B |

Exact suffix-exchange score reuse is cubic. Two-row screening chooses between direct forwards
and an exact segment programme, giving a whole structural-screening-pass O(N m^2 K^3) bound
without dropping candidates or changing the old screening/flat-neighbor rules. The quadratic
bin dependence is explicit; finite K grids can still favor the direct route and show quartic
trends. This does not bound entire search/count iteration counts. The
[2.7 complexity narrative](assembly_performance.md) gives the separate dependencies and scope.

Search budgets and incomplete families are reported explicitly. A search stopped at a
computational bound is not certified by the work that was left unperformed.

**Memory.** Completion's growing ball stores O(C r + K) support entries per site, plus
O(K^2) pair statistics; it no longer stores O(C K) allele rows and shell flags. Cross-fit EM retains
O(L K^2 + sum_l C_l r_l + N K^2) workspace, in addition to evidence. Cavity bin emissions retain
the existing O(N K^2 min(L,50)) arrays. The evidence view holds 37 N L bytes (float64 and float32 likelihoods and the observed
mask), shared by every worker. A join's packed binned emissions hold
8 S sum_j [K_j(K_j+1)/2] m_j bytes (2.38GB at S=320, K=30 and2,000bins), its exclusion forwards 8 S m bytes per excluded path (at
most K + 1 paths), its painting back pointers at most `AssemblyConfig.join.tensor_budget_mb = 256`
MiB per sample chunk, and the pool-saturation step O(B q K^2) move and beam entries at its top rung.
The linker keeps seven log emissions and seven weights per sample and proxy site. Exact pair
segments additionally reserve 16 m Q + 17 m P bytes plus index arrays per active numerical
thread when selected, Q=K(K+1)/2 and P the chunk's pair-query count; the direct route uses
only its small pair index. Concurrent seed calls reserve separately.

**Determinism.** Per-sample values are computed by one thread each and summed in sample order;
concurrent refits are evaluated in seed order; caches never change values; bootstrap streams are
seeded from the decision keys. Results therefore do not depend on thread counts, worker counts,
completion order or memory pacing.

## Interpretation and limitations

See [founder identity across joins](founder_identity.md) for the distinction between local allele
accuracy, long-range correspondence, penalized score margins and released identity breaks.

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
