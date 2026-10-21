# Discovery

Raw block discovery finds, independently in every 200-site block of a contig, a small panel of
binary founder haplotypes that explains the samples' genotype likelihoods at the block's discovery
sites. Per block it fits a hard, missing-aware fixed-K factorisation (K founder rows, each sample an
unordered pair of rows or unknown copies), searches over K and over panels with reversible moves
scored by a leave-one-sample-out cavity predictive, and releases the selected panel's alleles under
a capped pseudo-probability rule. It uses no pedigree, sample label or truth. Its selected hard
panel is the start of [local selection](local_selection.md); its released calls are a raw
diagnostic product.

## Inputs and outputs

| | |
| --- | --- |
| input | the contig's `evidence` record in the founder view (samples flagged by the contamination QC are unobserved; see [evidence](evidence.md)): per block its positions, discovery sites, genotype likelihoods and observation mask |
| product | one `DiscoveredBlock` (`discovery/release.py`) per informative block, `None` for an uninformative one; records `discovery/plan` and `discovery/blocks/<n>` on the contig file |
| identity | `identity/discovery`: `DiscoveryConfig`, the evidence record's identity and the code of the stage |

The stage has no user options; its constants are those of `DiscoveryConfig` (`discovery/config.py`),
listed in [configuration](../configuration.md).

## Current policy and release status

The adopted search uses an archive beam of six, parent-NLL-ranked replacement starts
with nominal budget 48 and a whole-class straddler ceiling of 96, residual groups with
nominal budget 64 and ceiling 128, and at most 256 proposal landmarks. These are the
2.3/2.4 defaults, not a private trial. The 96 replacement ceiling is distinct from the
historical optional escalation's nominal 96 budget. All observations, including G0 rows,
remain eligible for fitting and scoring subject to the evidence QC mask. Landmark
geometry changes proposals above its activation threshold; it is not an exact replacement
for dense geometry. Scientific identity means equal decisions and released uncertainty,
with continuous drift well below consequential scales, not equal floating-point bits.

Release 2.5 uses exact packed lifted-group counting, discovery diagnostics
and mode removal, and the accepted D8s complete proposal family. Bounded scientific
closures and their limits are recorded below; final full-pipeline validation remains pending.

## Ties and order (releases 2.1 to 2.3)

No discrete outcome of discovery may depend on floating-point rounding, on perturbations of the genotype
likelihoods far below any real difference, or on the order of the samples. Every block therefore has its tie quanta
(`maths/ties.py`, `BlockQuanta` from the block's null NLL M, the sum over samples of their genotype-oracle NLL plus
two unknown copies at every observed site, which bounds every panel's NLL): powers of two placed between the rounding
and noise motion of the compared quantities and their first genuine differences. Kind `nll` (at least
max(2^-17, 2^-30 M) nats) orders NLLs and cavity scores, kind `sample` (max(2^-23, 2^-36 M)) per-sample sums (A step,
data-birth excess), `h_step` (2^-18) the H step's allele sums, `improvement` (2^-14) the search's improvement tests,
`cell` (2^-23) per-cell release contributions and `probability` (2^-20) probabilities and masses. Values are grouped
into anchored tie classes (single-linkage chains of gaps at most the quantum over the exactly sorted values); every
order is (class, then exact secondary keys, then an exact content key), and a class wider than four quanta is ordered
by value inside (width guard). In release 2.3 the search scores and expands tied panel classes whole: the content key
labels a representative without selecting one biological answer from an equally supported set. A band with a fixed rule is
used only where a value can sit exactly on a threshold mathematically (an exact zero, exactly 1/2, a posterior split
exactly in half); continuous thresholds are compared as they are. The samples of a block are searched in the run's
canonical order (by sample name, `run_sample_order`; the evidence stage provides it). Population and cluster
reductions use accurately rounded expansion sums equivalent to `math.fsum`, and complete tied seed classes are
unioned before admission; neither a permutation nor a renaming can choose a seed or panel. Decisions taken inside a tie, width-guard and boundary-guard events and band decisions are counted in
the block's `SearchSummary.tie_events`.

The same evidence gives the same result on every CPU. Discovery evaluates log and exp with the C library's scalar
functions (Numba kernels), not with NumPy's float64 ufuncs, whose SIMD code paths NumPy chooses at run time from the
CPU (AVX-512 kernels differ from the scalar functions by an ulp in about 5% (exp) and 0.3% (log) of values); its other
arithmetic is element-wise or summed in a fixed order, and it makes no BLAS calls.

**Equal-distance links (release 2.2).** HDBSCAN (the K = 1 seeds and the residual-strand clusters) is HDBSCAN* on the
hdbscan package's mutual reachability and minimum spanning tree with every link of one weight merged at once, as
Campello, Moulavi and Sander (2013) and Campello et al. (2015) define it (edges of equal weight are removed
simultaneously): at every distance the hierarchy is the set of connected components of the links of at most that
distance, which no merge order changes (`residual_births.level_hdbscan`). A cluster splitting at lambda = 1/distance
into several components makes those with at least the minimum cluster size child clusters when at least two qualify;
otherwise it persists as its qualifying component and the other records fall out at lambda (two components: the
package's rule). Stability (sum of lambda - lambda_birth, summed exactly), excess-of-mass selection and labelling are
the package's; clusters are numbered by their smallest record. The package's binary dendrogram orders equal-weight
links by NumPy's unstable sort (CPU-dependent: actm3 chr8 block 204 under OpenBLAS Haswell kernels with NumPy's AVX-512
paths disabled), and release 2.1 by a fixed content key. Such ties are not confined to Hamming distances: with
min_samples > 1 (the K = 1 seeds) mutual reachability ties links through shared core distances.

**Tied resolutions carried forward (release 2.2).** A binary merge order decides what happens to records that fall out
together at a multi-way merge: it can pair them into clusters of their own, or let them fall out of the growing
cluster. Release 2.2 keeps both. The residual clustering's groups are every cluster of the level-wise HDBSCAN's
condensed tree (all density levels, selected or not) within the diameter bound (`max_cluster_hamming`) plus, as
alternatives, every maximal clique (at least two strands) of the graph
"distance at most `max_cluster_hamming`" on the strands HDBSCAN leaves as noise, the strands of clusters wider than the
bound, and the stragglers of multi-way merges (strands of at least two small components falling out where a big
component continues; they stay in their cluster as well). Every complete-linkage group at the bound and every pair-up
of tied stragglers within it, under any order, lies inside one of these cliques; complete linkage itself is not
used. Each group's consensus is a candidate and the block's cavity score decides between the panels they produce. The
budgets are order-invariant. A connected component is omitted only when its actual number of maximal cliques
exceeds `max_clique_enumeration`. Disconnected original subgraphs union their clique sets; disconnected complements
form complete joins, so their clique sets combine by Cartesian product. Exact child counts can establish an
oversized product without materializing it. Prime pieces use an exact compiled, pivoted Bron–Kerbosch traversal on
packed bitset adjacency rows. A first pass counts the actual maximal cliques, stopping at the limit plus one, so an
oversized component is omitted whole whatever its labels or traversal order; only an admitted component is
materialized, by a second identical traversal. Enumeration order determines work only, never the admitted
alternatives. `max_clique_calls` applies only to the legacy Python reference traversal, where it switches to
polynomial-delay bitset reverse search ([Comin and Rizzi (2015), section 3](https://arxiv.org/abs/1506.01082)); it
never truncates scientific output. The former worst-case count/work bounds do not reject components. No prefix of
traversal-selected cliques is released.

Residual groups are ranked by size, candidates by distinct sample support, and K = 1 seeds by cluster size.
Replacement starts (each K + 1 birth minus one row) follow the ranking specified under
[Replacement admission](#replacement-admission). For discovery's archive, proposal, score and expansion
budgets, classes are admitted whole until the nominal budget is reached.
A straddling class is admitted when the total remains at most twice the budget; beyond that it is omitted whole.
Strictly lower-ranked classes beyond the boundary are counted separately as `ranked_truncation`. Gauge maximum
cuts form one tied family: the previous effective budget is 4 x `max_cut_ties`, with the same twice-budget
allowance for that whole family. These bounds never choose a content- or sample-name prefix. A soft candidate
is a near-duplicate only of a panel row or a more-supported candidate; equally supported near-duplicates are
all kept, and exact duplicates are emitted once. Nearest-row ties use the union of their coverage tests.


**Released ambiguity.** A founder allele that no positive-weight carrier observes is unknown:
`DiscoveredBlock.latent_unknown` marks it (`latent` keeps a binary value only as the fit's initial value; `calls`
withholds it), and local selection, completion and export release '.' there. Release 2.3 cavity-scores every panel
of each K's leading NLL class. Members of the winning cavity class after the smaller-K prior that differ at an
observed allele are released together as `tied_panels` (`tied_panel_k`, `tied_panel_nll`, `tied_unknown`); local
selection takes them as proposals, so neighbouring blocks decide. Raw calls are conservative across the complete
alternative set. An omitted straddling class or whole-component gate sets `SearchSummary.search_incomplete`.
The `omissions` entries carry `kind` (`tie_class_at_budget` or `ranked_truncation`), cause, count, scope, and
`count_kind` (`exact` or `lower_bound`). Ordinary ranked truncation has per-cause statistics but does not set
`search_incomplete`. These are computational limits of the documented search, separate from biological
ambiguity; valid incumbent calls stay eligible for local selection. Only when no scored incumbent exists is the
block `unresolved`, with an empty latent/call panel. Counts are policy events, not globally unique candidates.


**Convention reversal (release 2.2).** With `HAPLOTYPES_TIE_REVERSE=1` (maths.ties.reverse_conventions) every remaining
arbitrary convention runs in reverse: content-key and index orders inside tie classes, the sample-name order, the
Gauss-Seidel founder order on usage ties, the binary default state at exactly 1/2, argmax ties, the gauge max-cut code
order, the proposal order before the expansion budget, dominant-partner ties and the mean-field start order
(`discovery/conventions.py`; the diagnostic variable `HAPLOTYPES_TIE_REVERSE_KEEP` keeps named conventions at their default). Stated preferences (priors)
are not reversed: a smaller K, keep-current inside a tie (incumbent allele, first fit unless better by a quantum,
incumbent search state), the conservative release band at an exact zero, more supporting samples first, more observed
sites first and fewer unknown copies first; nor canonical labels (a panel's lexicographic row order and key).

**Fractional tied-state evidence (release 2.3).** Let T_i be every state within the existing
`sample` cost band in the category with the fewest unknown copies. Sample i contributes weight
1/|T_i| to each member, uniformly over unordered states. These weights are a symmetric allocation
of unresolved evidence, not a calibrated posterior. They do not use the heterozygote multiplicity
of the cavity prior. For example, T = {(i,i), (i,j), (j,j)} contributes one sample's state mass
and one expected copy to each founder; the homozygous term contributes one genotype cost, not two.

The H step averages the existing capped state costs, reading partners under the existing
Gauss-Seidel schedule. The cavity founder field receives weighted homozygous/unknown-copy unaries
and heterozygous edges; hold-out removes every affected factor and each state's fractional count.
Founder support counts distinct samples with positive membership. Release applies the cell band
to each unweighted contribution before weighting, and counts each sample at most once per
founder/site. An allele is observed whenever any positive-weight carrier observes it.

This policy applies to discovery and every local-selection bank fit, score and release; there is
no `tie_meet` gate. The meet remains only a compact pair summary and a gauge-proposal heuristic.
Multi-state memberships are retained as compact offsets and state IDs, remapped with founder and
sample axes. Temporary dense membership masks have a 32 MiB frontier chunk bound. Cavity solves
are reused only for identical complete tied-state memberships and identical per-site likelihood
triples; membership lists are compared in full. Such cases have identical founder-support deletion,
weighted unary and edge patches, starts and held-out prior deletion. Each original sample keeps
its own predictive and failure-count contribution. Grouped results are checked bit for bit against
independent ungrouped solves and a slow field-rebuild reference.

The fractional evidence allocation leaves the all-state switching-Viterbi NLL, `unknown_slots`,
caps, penalties, quanta, stopping rules, cavity functional, K prior and set code unchanged. The
separate D12 policy below changes the approximate predictive functional only when distinct
training-ELBO-tied mean-field endpoints remain. Fitted panels and cavity values can
change because evidence that release 2.2's meet discarded now reaches the founders. No new
monotonicity or EM claim follows: A, H and the reported NLL still use different functions.

## Block evidence and missingness

A block is fitted on its discovery sites only (see [evidence](evidence.md#discovery-sites-and-blocks)).
A block is informative when at least one discovery site is observed in at least one sample;
uninformative blocks get no panel. Before any seed, fit or score, every unobserved cell is set to
exactly 1/3, and missing data stays neutral throughout: unobserved cells have zero cost in the
fixed-K fit and no unknown-copy penalty, a constant predictive (the same for every state, so it
cancels across K) in the cavity score, and no vote in the release. A sample without any observed
discovery site in the block is assigned two unknown copies and is left out of the cavity score.

Discovery pools the compatible evidence of all carriers of a founder at each site, so a founder
allele can be called where some carriers are missing and others are informative. Positions without
sufficient support stay unknown (-1); missingness never counts as an allele match.

## Fixed-K model

A panel H is K distinct binary rows over the block's L discovery sites. Each sample takes one of
(K + 1)(K + 2)/2 diploid states: an unordered pair of rows (i, j), i <= j; a row with an *unknown
copy*, whose allele is chosen freely per site at a fixed penalty; or two unknown copies. With
GL the sample's normalised likelihoods at an observed site and
lambda = `DiscoveryConfig.model.unknown_copy_penalty = 0.5` nats per unknown copy, the per-site
costs are

- row pair (i, j): -log max(GL[H_i + H_j], 1e-12);
- row k with an unknown copy: -log max(GL[g]) over g in {H_k, H_k + 1}, plus lambda;
- two unknown copies: -log of the largest GL, plus 2 lambda.

Sites are grouped into bins of `DiscoveryConfig.model.snps_per_bin = 10` sites (one bin per site
when a block has at most that many discovery sites). A sample's negative log likelihood (NLL) is
minus its best binned switching-Viterbi log likelihood: its diploid state may change between
adjacent bins at a cost of `DiscoveryConfig.model.switch_penalty = 10.0` nats. The panel's NLL is
the sum over samples.

A fit alternates two steps from a start panel:

- **A step**: each sample's whole-block tied-state set, by its summed (uncapped) bin
  emissions: among the states whose cost is within the block's `sample` quantum of the cheapest, the
  category with the fewest unknown copies (row pairs, then a row with an unknown copy, then two
  unknown copies; a stated preference), and uniform evidence over all its tied states (see
  "Ties and order");
- **H step**: Gauss-Seidel founder-bit updates for the fixed assignments, founders in decreasing
  order of expected-copy usage; for founder k at site l the costs of allele 0 and 1 are averaged
  over each carrier's tied states and summed over carriers
  (homozygous carriers, carriers with a known partner, carriers with an unknown copy), each cost
  capped at the cell's two-unknown-copies cost; the allele changes only when the two sums differ
  by at least the block's `h_step` quantum.

The fit stops at a fixed point (an H step that changes nothing) or after
`DiscoveryConfig.model.max_iterations = 50` iterations. Because the A step's ranking cost, the H
step's capped cost and the binned NLL are different functions, the deterministic map can cycle; a
cycling fit returns the cycle's lowest-NLL state (NLLs within the `nll` quantum are tied and keep the order in which
the fit first visited them); it reports the iteration limit and is never certified as a fixed point, so a data birth
refits it. Panel NLLs are exactly rounded sums (`math.fsum`).
Panels are kept in canonical form (rows in lexicographic order, assignments remapped); of two fits of
the same panel the first is kept unless the other is lower by more than the `nll` quantum.

**Complete usage-schedule families (D3).** Every discovery and candidate-bank fit canonicalizes
its start's founder rows and evaluates two consistent schedules: decreasing positive expected-copy
usage, with exact usage ties visited in increasing or decreasing canonical founder index. A
trajectory uses one schedule throughout. The opposite trajectory is evaluated only if the first
encounters an actual positive-usage tie; otherwise the trajectories coincide and only one is run.
This is a family of two documented schedules, not an enumeration of every possible update order.
Distinct endpoints reach the existing selectors, and tied endpoints remain alternatives. A
collapsed endpoint is refitted at its lower K within the same original start's family.

Both trajectories share exact A(H) records and identical untied H transitions. An H transition
can also be shared across tied schedules when every equal-usage group has no internal heterozygote
membership dependency, or when the entire H sweep changes no allele. In the first case the tied
founder updates commute; in the second, each update reads the original panel and retains its allele,
so induction gives the same result in the opposite order. Both conditions prove the identical next
panel. The usage tie remains marked, and the opposite trajectory still runs: its later path may
diverge. Other tied transitions, complete-fit caches and fixed-point certificates retain schedule
identity. A shared transition does not certify an unexecuted trajectory, and a family certificate
requires both schedules. The iteration and cycle rules above remain unchanged. The assignment-only
interface evaluates C(H) without H work or schedule multiplicity.

## Cavity score

The search compares panels, including panels of different K, by a regularised fixed-assignment
leave-one-sample-out predictive score (`discovery/cavity.py`):

    score(H) = sum_i log sum_s pi_-i(s) prod_l p_bar(GL_il | s)  -  log K  -  log(K + 1)  -  log C(2^L, K)

- **State prior.** pi_-i(s) = (n_s - w_i(s) + alpha beta_s) / (N - 1 + alpha) is the
  Dirichlet-multinomial predictive of the state counts n_s of the panel's assignments with sample i's
  own fractional count w_i(s) removed; alpha = `DiscoveryConfig.cavity.state_concentration = 0.5`. A sample whose A step
  held several tied states T counts 1/|T| at each of them (and removes 1/|T| from each when held out;
  release 2.2). The prior masses
  beta give the row-pair states 1 - omega (a heterozygous pair twice the weight of a homozygous one),
  each row-with-unknown state omega (1 - delta) / K and the two-unknown state omega delta, with
  omega = `DiscoveryConfig.cavity.unknown_prior_mass = 0.02` and
  delta = `DiscoveryConfig.cavity.double_unknown_fraction = 0.25`.
- **Founder alleles.** q_l^(-i) is a mean-field posterior over the K founder alleles at site l
  given every other sample's assignment and genotype likelihoods: a pairwise Markov random field
  with a uniform allele prior, unary terms from homozygous carriers and carriers with an unknown
  copy, and edge terms from heterozygous row pairs, all weighted by 1/|T_i|. Holding out sample i
  removes all its weighted factors using saved unary/edge lists, then restores their exact values. Each held-out
  solve runs three starts (the panel's alleles, their complement, all one half; founders without
  other carriers stay at one half) for at most `DiscoveryConfig.cavity.mean_field_max_iterations = 100`
  sweeps to tolerance `DiscoveryConfig.cavity.mean_field_tolerance = 1e-10`. If any start converges,
  only converged endpoints are eligible; otherwise all endpoints are eligible and the failed solve
  is reported. Retain every eligible endpoint within the block's existing `nll` quantum of the
  maximum training ELBO. Deduplicate exactly equal founder-probability vectors, then connect
  endpoints whose maximum absolute coordinate difference is at most the configured mean-field
  convergence tolerance (1e-10 by default). Connected components define endpoint-equivalence
  classes without choosing a representative by start order. This reuses the solver's existing
  convergence scale; no new threshold is introduced. A founder update whose logit is within the
  same quantum of zero sets its allele probability to
  exactly one half, preserving the solver's treatment of symmetric evidence.
- **Predictive mixture (D12).** For held-out sample i and site l, let C_il be those endpoint
  classes. Under each endpoint q, p(GL_il | s, q) averages sample i's likelihood over independent
  founder alleles for state s; an unknown copy carries ALT with probability 1/2. First average raw
  predictive likelihoods over the distinct vectors within each class, then average those class
  predictions uniformly:
  p_bar(GL_il | s) = sum_{c in C_il} [sum_{q in c} p(GL_il | s, q) / |c|] / |C_il|.
  Apply the existing probability floor only after this mixture, before taking its logarithm;
  combine sites and the unchanged state prior as above. Multiple numerical endpoints within one
  class cannot give that class extra weight. Founder marginals are never averaged first.
  Eligibility and class membership use the training solve, and held-out data never choose mixture
  weights. This deliberately changes the approximate predictive functional in training-ELBO-tie
  cases; with one exact endpoint it reduces to that endpoint's predictive. It does not maximise
  held-out prediction over starts. Eligibility, the `nll` quantum, state prior, K prior and set code
  are unchanged.

Exact vector equality alone failed an endpoint-identity probe: on four K30 blocks under iid12,
iid9 and value-consistent noise (12 block/variant scopes), 1,456–2,021 of 16,000 sample-site solves
per scope changed their exact endpoint count. All 12 scopes retained the same count of classes at
the existing 1e-10 convergence tolerance. That evidence motivates the equivalence scale above;
it is not a claim that the full pipeline's noise gate has passed.

This bounded probe does not establish global invariance of endpoint multiplicity. Later parent7
K30 diagnostics under iid9 changed equivalence-class histograms in six of 200 blocks. Across
209,408,000 sample-site entries accumulated over scored panels, aggregate counts changed by
-50 in the one-class bin and +50 in the two-class bin. These are net histogram differences,
not fifty identified per-solve transitions. All 200 complete biological outcomes and selected
objective classes were unchanged. The existing equivalence scale is retained; no additional
threshold is introduced.

- **Regularisers.** -log K - log(K + 1) is a telescoping prior on K (the terms 1/(K(K + 1)) sum to
  one), and log C(2^L, K) is the code length of an unordered set of K distinct binary rows of
  length L.

The score is a pseudo-score, not calibrated evidence: the panel and the assignments it holds out
from were themselves selected on the full data.

## Reversible search over K

The search (`discovery/search.py`, moves in `discovery/moves.py`) keeps an archive of every fitted
complete panel (distinct rows), one entry per canonical panel with its first fit (replaced only by a
fit lower by more than the `nll` quantum). Per K the archived panels are ordered by tie classes of
NLL, then canonical key. Every member of the leading NLL class is cavity-scored, memoized by panel key.
The winning cavity class, the existing wide-class rule and the smaller-K prior determine the retained alternative
set. A representative labels this complete set; panel bytes cannot choose which tied panel supplies evidence.
There is no K grid and no K cap besides the identifiability ceiling min(2 N_active, 2^L), N_active
being the samples with depth in the block.

**Starts.** The K = 1 starts are pooled rows of the largest HDBSCAN clusters (at most
`DiscoveryConfig.search.data_seed_count = 8`, minimum cluster size
`DiscoveryConfig.search.seed_min_cluster_size = 2`) of the samples with depth, clustered by their
expected genotype agreement (summed in one fixed order, exactly symmetric); a cluster's mean ALT
fraction within the `probability` quantum of 1/2 takes the population ALT fraction. Distinct seed rows are unioned
within each cluster-size class before the seed-count cap; duplicate rows consume one slot. With no cluster seed
class, all samples with the most observed sites and best decisiveness class contribute their distinct dosage rows.
The whole-class admission rule above applies at the seed cap; no population row substitutes for
an omitted class. Starts are fitted as complete usage-schedule families, then retained by whole NLL
classes at `DiscoveryConfig.search.data_start_beam_width = 6`.

**Dosage families (D8a).** At each sample/site, use the full winning genotype-likelihood class at
`probability` quantum 2^-20, with the common chained-class rule. Its minimum and maximum dosages
form the low and high dosage vectors. Their distinct rows contribute to the initial fallback,
direct data births and every founder-subtraction birth; rows and complete starts are deduplicated
before the existing whole-class budgets. No REF/ALT or dosage prior is added. The population-half
hardening rule (D8s) is separate: this construction alone is not a proof of general REF/ALT relabeling
invariance.

**Population-half families (D8s).** When population ALT is within the existing HALF_BAND
of one half, both coherent population hardenings enter as one complete proposal family.
Every tied site takes the first convention together or its complement together; this is
not enumeration of all independent sitewise completions. K1 cluster/dosage starts, data
births and soft-residual hardening union and deduplicate distinct rows before existing
fitting and whole-class admission. Reversing D8s only orders the same family. Objectives,
priors, weights, thresholds and budget values are unchanged; existing release carries
co-winning fitted alternatives. More starts can change active search paths, omission
counts and released calls, even though the configured budgets and admission rules stay fixed.
This does not restore a reversed second search or prove general REF/ALT invariance.

The accepted W3 gates preserve inactive products in 4/4 cases and preserve the complete
released family under D8s hook reversal in 5/5 active cases. Four frozen depth-0.1 fixtures
have small mixed truth differences against the shipped default, judged not materially worse
by the coordinator; no accuracy improvement is claimed. In the N320 fixture, common-called
errors change from 19 to 20 of 508 while wrong-or-missing cells fall from 697 to 694 of
1,200 truth cells. The std fixture adds four calls and one error while retaining all 112
common calls at 15 errors; wrong-or-missing falls from 1,103 to 1,100 of 1,200. These are
bounded founder-truth measurements, not downstream or full-pipeline acceptance. A consequential
second soft-residual hardening was not reached in those fixtures; that path has source-review
coverage, not an empirical released-effect gate. Search cost has no general twofold bound.

**Expansions.** Each expansion takes the beam of the selected K, its
`DiscoveryConfig.search.beam_width = 6` first archived panels.

- *Ordinary phase.* Freeze the next complete unexpanded NLL class. Every member proposes deaths, gauge
  rewires and data births before the union is merged and rescored. If that union did not improve the score by more
  than the block's `improvement` quantum, birth-delete replacements of the union of births follow.
- *Residual phase.* Once every beam panel was expanded ordinarily, the next unexpanded whole NLL class proposes
  soft-residual births, then replacements, under the same atomic schedule. With no improvement, bridge every tied,
  data-selected K + 1 parent through its data and soft births to K + 2. One parent's improvement cannot suppress
  its tied peers' neighbourhoods.

**Moves.** Each move proposes start panels; every start is fitted as a complete usage-schedule
family. Its assignment-only fit and every complete endpoint enter the archive (an endpoint that
collapses rows re-enters as a start at the lower K). Data births likewise refit parents to complete
families. Parent/sample pairs share one excess-and-coverage ranking and one existing seed budget;
there is no per-endpoint budget multiplication. Complete starts are globally deduplicated within
the admitted class, and every admitted child's endpoint family is completed before testing the
class-level child target.

| move | K | proposal |
| --- | --- | --- |
| death | K - 1 | every one-row deletion |
| gauge rewire | K | for a maximum cut of the meet-summary graph (a proposal heuristic only; rows linked by the number of meet pairs containing both), flip the sites where both sides are internally constant and differ: per site to the better allele (by more than the `nll` quantum) for the samples within one side, and all such sites at once |
| data birth | K + 1 | the samples with the largest excess NLL over their sitewise genotype oracle (tie classes of kind `sample`; an all-unknown sample's excess is its exact value 2 lambda n_obs), then more observed sites, then canonical order, propose K subtraction seeds clip(dosage - row, 0, 1) and their dosage row; tied samples form groups consumed whole: starts are collected group by group up to the attempt budget max(16, 2(K + 1)) x `data_birth_attempts_factor = 3` and fitted group by group until (K + 1) x `data_birth_children_factor = 3` distinct complete children exist (tests at group boundaries); distinct starts are unioned within the class before admission; a class crossing `data_birth_tie_budget_factor = 2` x the budget is omitted whole (limit `data_birth_tie_budget`) |
| soft-residual birth | K + 1 | consensus rows of residual strands (below), hardened at 1/2 (within the `probability` quantum of 1/2 the population ALT fraction decides) |
| replacement | K | a K + 1 birth with any one row deleted; see [Replacement admission](#replacement-admission) |

Deaths and gauge rewires together have nominal proposal budget
`DiscoveryConfig.search.max_proposals_per_expansion = 128` starts, as do soft births;
replacements have nominal budget `DiscoveryConfig.search.max_replacement_children = 48`.
The whole-class allowance admits a straddling class through twice its nominal budget. Death and gauge proposal families are
admitted whole, in that search-priority order, with no byte-prefix truncation. Assignment-only fallback births
are cut by whole NLL classes. Gauge rewires use exact
maximum cuts up to `DiscoveryConfig.search.exact_cut_max_k = 12` rows (by connected components:
bipartite components by 2-colouring, the others by exhaustive enumeration of the 2^(m-1) cuts of
a component of m rows); above that, degree-ordered local searches from every singleton and from an
alternating start. With `DiscoveryConfig.search.max_cut_ties = 4`, the effective gauge-family budget is 16;
complete tied families are admitted through 32 cuts. A larger family is omitted whole and reported.

**Soft-residual births** (`discovery/residual_births.py`). Each sample's posterior over unordered
row pairs (uniform prior over the hard pairs) gives every row an expected copy count; each row with
at least `DiscoveryConfig.residual_births.min_soft_responsibility = 0.25` expected copies in a sample
is subtracted from it, leaving a residual strand whose ALT probability at an observed site is
P(GL | subtracted allele + residual allele) normalised over the residual allele. Residual strands
called at `DiscoveryConfig.residual_births.residual_call_probability = 0.8` are clustered by
missing-aware Hamming distance (level-wise HDBSCAN, plus maximal cliques of its noise and tied stragglers as
alternative groups, see [Ties and order](#ties-and-order-releases-21-to-23); diameter at
most `DiscoveryConfig.residual_births.max_cluster_hamming = 0.1` of the sites), over all strands and
within each (subtracted row, dominant partner) route. A cluster's consensus pools its strands'
responsibility-weighted log odds with one capped vote per sample; it is emitted when at least
`DiscoveryConfig.residual_births.min_unique_sample_support = 2` distinct samples and
`DiscoveryConfig.residual_births.min_effective_sample_support = 1.5` effective samples support it,
at least `DiscoveryConfig.residual_births.min_candidate_known_fraction = 0.8` of it is called at
`DiscoveryConfig.residual_births.candidate_call_probability = 0.9`, and it duplicates neither a
panel row nor an earlier emission. A strand's log odds are log(residual ALT mass) - log(residual REF
mass), within the bounds [log 2^-1022, log((1 - 2^-52) / 2^-52)] (about -708 and 36), computed
from the masses rather than from the rounded probability. Each original residual enters every
partner route whose mass is within the existing `probability` quantum times max(1, the global
maximum partner mass) of that maximum (D10p). It remains one global observation and appears once,
with its original responsibility weight, in each eligible route. Routes are alternative proposal
pools, not additional observations or votes. Identical member sets share one consensus computation;
exact candidate duplicates are removed before the unchanged whole support-class budgets. The
reporting-only partner label cannot choose a route. A record weight or an
effective support within the `probability` quantum of its threshold counts as at the threshold
(symmetric evidence splits posteriors exactly in half). Consensus rows are emitted with more distinct
supporting samples first, then in the order of their hardened rows, then source order (so of two
near-duplicates the better supported one is kept).

### Replacement admission

The adopted policy retains fitted K + 1 births' NLL ranking. Union and deduplicate all
one-row-deletion starts within each complete parent NLL class, and remove duplicates already
seen in earlier classes. A repeated start consumes one slot, in its earliest parent NLL class.
The ordinary search has nominal budget R = 48 unique replacement starts and admits complete
classes until it reaches that budget. A straddling class is retained only if adding it keeps the
cumulative total at most 2R = 96; otherwise the entire class is omitted. Optional
escalation and reversed-convention second search have been removed. The historical
2.3/2.4 escalation nominal budget of 96 is not a current option. The ordinary 48/96
whole-class policy and parent-NLL ranking are unchanged.
A replacement start's own assignment-only NLL C(H) does not change admission priority.

The evaluated alternative globally deduplicates starts and ranks whole classes by their own exact
assignment-only C(H), using bounded batches without pruning and an exact scalar-score memo. It is
not the adopted replacement ranking. The isolated own-score arm failed stress truth; parent
ranking at the provisional parent7 nominal 24/48 budgets also showed a K30 truth regression.
These historical experiments motivated the adopted doubled-budget policy; neither is a
final-pipeline acceptance result.

In the historical isolated doubled-budget K30 discovery trial, the policy reduces called founder errors from 361 to 331,
missing truth founders from 900 to 888, missing-inclusive errors from 223,419 to 221,609, and
blocks with computational omissions from 56 to 3 of 200. All seven founder metrics improve
on 2.2 in that trial; matched warm wall time increases 11.2%. These are discovery-stage results,
not final released accuracy. The adopted A256 component campaign subsequently retained all seven truth metrics in 38
accuracy cells and passed 56 invariance controls. Final full-pipeline coverage remains
incomplete as recorded in [performance](../performance.md#validation-coverage-at-freeze).
Larger-budget sensitivity remains a measured limitation.

Both rankings define bounded searches. The selected objective need not improve monotonically with
a larger budget: newly found leading NLL classes can change which panels receive cavity comparison.
Budget sensitivity is a measured limitation, distinct from biological ambiguity.

**Stopping and limits.** The search stops when the selected K's beam is exhausted in both phases,
or at the expansion backstop. Backstops scale with K_rep, the largest K represented in the archive:
nominally 32 + 8 K_rep expansions (`DiscoveryConfig.search.expansion_backstop_base = 32`,
`DiscoveryConfig.search.expansion_backstop_per_founder = 8`) and nominally 96 + 12 K_rep cavity scores
(`DiscoveryConfig.search.exact_score_backstop_base = 96`,
`DiscoveryConfig.search.exact_score_backstop_per_founder = 12`). The expansion backstop is tested before a whole
parent class. Both backstops admit a straddling class through twice their nominal count, then stop at the class
boundary. An overlarge leading NLL class is omitted whole; the last completely scored incumbent remains valid.
Cached scores do not justify a partial class. Limits constrain work, not the number of founders. The stop reason
is `generated_neighbourhood_local_optimum`, `incumbent_neighbourhood_search_limited` or `expansion_backstop`.
Omitted classes/gates have `*_tie_class_at_budget` causes; ranked tails have `*_ranked_truncation` causes and
are excluded from the search limits. Mean-field nonconvergence is reported separately. Heuristic maximum cuts
above K = 12 are reported as `heuristic_gauge_cuts` and are not a computational omission.


**One search policy.** Release 2.5 removes optional escalation and the reversed-convention
second search. Custom Python callers must remove the former SearchConfig mode
arguments; there is no replacement switch. These internal constants were never supported
TOML keys. Explicit A=0 dense geometry and acceptance-only convention-reversal diagnostics
remain available. Historical `escalated` and `conventions` summary metadata still decode;
new runs record False and 1 respectively.

**Certification.** A selected panel is certified when it was expanded in both phases without any
limit for every tied winning panel, every K its moves generated is scored, and no move improved the score: a certificate of a
local optimum within the generated neighbourhood, not of a global one.

## Interpreting search status

| Term | Meaning and consequence |
| --- | --- |
| Ranked truncation | Strictly lower-ranked candidates beyond the admitted boundary are not searched. Counts describe the bounded policy; they do not set `search_incomplete`. |
| Incompleteness | An entire tied boundary class or a computational gate is omitted. An established incumbent remains usable; this flag is not an error rate or biological ambiguity. Counts may include repeated cached visits, not distinct alternatives. |
| Unresolved | No initial incumbent can be established because all available initial classes are omitted. The block has an empty latent/call panel. |
| Non-convergence | A numerical solve did not satisfy its convergence criterion. Fixed-K cycles retain their best cycle state; cavity scoring prefers converged starts and uses finite endpoints if none converge. The block flag concerns any scored candidate; separate counts now distinguish the complete winning set from other scored panels. |
| Certification | A local statement about all tied winners and their generated, scored neighbourhood, subject to the documented numerical and search limits. It is not a global optimum, ancestral-founder truth or accuracy guarantee. |

These conditions are distinct from evaluated ambiguity, where complete competing answers
remain tied and the release rule preserves their uncertainty. A heuristic gauge family above
K=12 is reported separately and does not by itself set incompleteness. The saved 2.3 TroMau
records have 390/9,979 incomplete blocks and 488/9,979 blocks with some scored-candidate
mean-field non-convergence; selected-panel failure cannot be recovered from those flags.

## Release rule

The selected panel's hard rows are *latent*: they are complete at the discovery sites and are what
local selection fits from, but they are not calls. Alleles are released per founder k and site l
with the selected assignments held fixed (`discovery/release.py`). Each observed carrier of k
contributes its capped cost difference min(c0, cap) - min(c1, cap), with the H step's costs and
cap (homozygous carriers g = 0 against 2; carriers with a known partner j, H_j against H_j + 1;
carriers with an unknown copy, the row-with-unknown costs). Apply the cell band before the
1/|T_i| state weights; sum weighted contributions and count a sample only once per founder/site
when any member contributes informative evidence. With E_kl the sum of these
contributions, the capped pseudo-probability of ALT is p_kl = 1 / (1 + exp(-E_kl)), and the
informative supporters are the carriers whose contribution exceeds the block's `cell` quantum in
absolute value, in either direction (symmetric evidence gives exactly zero). The
allele is released as 1 when p_kl >= `DiscoveryConfig.release.min_call_probability = 0.85`, as 0
when 1 - p_kl >= 0.85, in both cases only with at least
`DiscoveryConfig.release.min_supporters = 2` supporters; otherwise it is -1. Sites that are not
discovery sites are -1 with p = 0.5. The same rule releases the source endpoints of local
selection's candidate bank, whose calls start the path search.

## Search summary

Each `DiscoveredBlock` carries a `SearchSummary` (diagnostics; nothing downstream decides on it):

| field | meaning |
| --- | --- |
| `k_values`, `k_log_scores` | represented K values and the best cavity score among each K's completely scored leading NLL class |
| `natural_k_ceiling` | min(2 N_active, 2^L) |
| `expansions`, `data_starts`, `exact_scores`, `exact_score_cache_hits` | work counts |
| `residual_calls`, `residual_rows` | soft-residual birth calls and the novel rows they emitted |
| `limits` | reported limitations, in first-occurrence order |
| `heuristic_gauge_cuts` | some expanded panel had K > 12 and used heuristic maximum cuts |
| `stop_reason`, `certified` | why the search stopped; the local certificate above |
| `tie_quantum` | the block's `nll` quantum |
| `archive_max_k` | largest K ever added to the archive, including subsequently pruned panels; unavailable in old records |
| `wall_seconds`, `cpu_seconds`, `phase_seconds` | completed-block wall/process-CPU and phase buckets, with parallel geometry in a mixed envelope; unavailable in old records |
| `selected_mean_field_failures` | held-out site solves with no converged start, summed over the complete winning fitted set including tied alternatives |
| `other_mean_field_failures` | such failures in uniquely scored cached panels outside that winning set, including pruned panels; cache hits do not duplicate counts |
| `escalated`, `conventions` | historical mode metadata retained for decoding; new runs use False and 1 |
| `ambiguous`, `tied_alternatives` | multiple complete panels remain in the winning cavity class after the wide-class rule and smaller-K prior, and how many alternatives |
| `search_incomplete`, `omissions` | an oversized straddling class or whole gate was omitted; each entry gives `kind`, cause, count, scope and `count_kind`; `ranked_truncation` entries do not set incompleteness |
| `unresolved` | no incumbent could be established because every available initial candidate class was omitted; the block has an empty latent/call panel |
| `score_margin` | the selected score minus the best other represented K's score |
| `tie_events` | per decision kind: decisions, decisions taken inside a tie, width-guard and boundary-guard events (diagnostic; not an outcome) |

Residual-route diagnostics include `discovery.residual.partner_routes` (original observations and
multiroute observations), `discovery.residual.partner_route_memberships` (all memberships and those
beyond one per observation), and `discovery.residual.partner_route_pools` (distinct route pools).
They report computational routing; they do not withhold calls. Landmark/clique omission
scopes also record generating K and phase without changing their kinds or counts. Expansion
and score backstop reports count affected blocks using actual stop reasons/omissions, with
ordinary ranked truncation and omitted tied classes reported separately.

Block memory estimates are prepared before dispatch. A cheap lookup supplies the estimate
while each task is timestamped before admission. Block wall starts when a pool task is
drawn, before memory admission, and ends after allele release; estimate preparation is
outside that timer. A direct call starts at in-memory entry. `idle_admission` covers dispatch preparation,
memory admission, transport/startup and the initial thread claim, excluding waiting for an
undispatched pool slot. CPU covers the worker's numerical threads, not child processes.
Parent timing buckets are `fitting`, `landmarks`, `clique_counting`, `lifting_budgeting`,
`cavity` and `idle_admission`; fitting includes otherwise unseparated search/release
orchestration. Parallel residual geometry is one `lifting_budgeting` envelope with mixed
worker work, including landmark selection and clique counting. Worker timing contexts
are not propagated: zero serial geometry buckets do not measure zero worker cost.
Auxiliary task-wall sums can overlap and must not be added as exclusive elapsed time.
Heap trimming, checkpoint writes and unfinished killed blocks are excluded. Historical
unavailable fields decode as None or an empty phase mapping, not fabricated zeros.

## Records

| key | content |
| --- | --- |
| `identity/discovery` | stage identity of the contig's discovery records |
| `discovery/plan` | the number of blocks B of the evidence partition |
| `discovery/blocks/<n:06d>` | chunk n, in write order, of up to 64 completed blocks in completion order: their indices and encoded `DiscoveredBlock`s (`None` for uninformative blocks) |

A `DiscoveredBlock` holds the block index and first site, the block's positions and discovery
sites, `latent` (`int8[K, L_kept]`, canonical row order), `calls` (`int8[K, L]`),
`alt_probability` (`float64[K, L]`), `support` (`int32[K, L]`), `unknown_mass` (the selected
assignment's unknown copies over the 2 copies of every sample with depth), `nll` and `search`. No
genotype likelihoods are stored: blocks refer to the evidence record by site span. A contig is
complete when every block index is recorded; readers are `discovery.records.load`, `complete`,
`progress` and `founder_panels` (released calls of the informative blocks, for evaluation).
Completed blocks and their search diagnostics are buffered and committed together in the
scientific chunks. The writer flushes at 64 pending blocks, or at the next block completion
when about 30 seconds have elapsed since the last flush; it also flushes the final partial
chunk. The elapsed-time check is completion-driven, not a background timer: a long gap
between results can exceed 30 seconds. An interruption preserves committed chunks;
uncommitted blocks are recomputed on resume. Diagnostics require no separate per-block
write or fsync. The schema remains 1; normal source/configuration identity rules apply.

## Implementation and cost

| module | role |
| --- | --- |
| `discovery/stage.py` | pool over blocks, one contig ahead on a helper thread; `discover_block` for one block in memory |
| `discovery/fixed_k/` | workspace (evidence-only tables and fit caches), A step, H step, frontier fitting of many starts |
| `discovery/cavity.py`, `discovery/cavity_kernels.py` | cavity score |
| `discovery/moves.py`, `discovery/residual_births.py`, `discovery/search.py` | proposals and the reversible search |
| `discovery/release.py`, `discovery/records.py` | release, product, records |

Blocks are independent. One process pool serves every block of every contig, and the next contig's
evidence is prepared on a helper thread while the current contig is discovered. Within a block, the
fixed-K fits of many starts advance together as frontiers over the two memoised schedule maps
(equal A states and untied transitions are evaluated once) and use up to `discovery.fixed_k.fit.FIT_MAX_THREADS = 16`
threads, claimed at phase boundaries; the cavity kernel runs held-out samples in parallel.
Discovery alone enables CPU handoff to memory-admitted blocks waiting for their first CPU.
At the owner's existing apply/claim boundary, borrowed claims can shrink to a fair share
including those waiters, with the Numba team resized in the same call. Pure share queries
retain running claims; normal growth resumes as waiters drain. Other stages retain the
default policy, and memory admission and reservations are unchanged. Small residual work runs inline in source order with shared caches, avoiding
executor setup, per-pool cache copies and repeated memory-accounting calls for tiny work.
Pools execute inline when all use dense geometry, only one pool is present, or only one
budget thread is available. Component jobs with at most 4,096 estimated traversal steps
and at most 1 MiB scratch also run inline within the block workspace. Storage above
1 MiB retains an explicit counted reservation. These thresholds affect execution only.
Substantial independent pools and ordinary or lifted
clique components use the concurrent path when spare budget threads can help. Estimated
work orders parallel tasks largest first; counted output sizes govern large-storage
reservations and materialisation. There is no fixed core split or named-contig scheduling.
Read-only cache snapshots feed parallel tasks; their deltas, omissions and sources merge
in original source order. Execution thresholds select scheduling only, preserving admission,
groups, complete classes and omissions. Every sum over samples or sites runs in a fixed order, so results do not depend on
thread counts, batch composition or the order in which blocks complete.

Cost in the founder count K, samples N and discovery sites L of one block:

- **Fixed-K fit**: the A step evaluates (K + 1)(K + 2)/2 states over L/10 bins per sample,
  O(N K^2 L); the fractional H step visits the tied-state memberships of each carrier. With M
  total memberships its work is O(L M + K L), reducing to O(N L + K L) for one state per sample.
  Each of the at most two schedules runs at most 50 iterations; exact A work and certified identical
  H transitions are shared between them, while both tied trajectories remain represented.
- **Cavity score of one panel**: per site and held-out sample, mean-field sweeps over K founders
  and at most K(K - 1)/2 active edges, and a predictive over O(K^2) states: O(N L K^2) per sweep,
  with exact reuse of equal held-out cases.
- **Proposal geometry**: K1 pools and residual pools larger than A=256 use the bounded
  landmark construction described [below](#bounded-proposal-geometry-in-23). A residual
  pool with R records has sorting and descriptor terms O(R L log R + A R log R),
  distances O(R A ceil(L/64)), and hierarchy/clique work F(A). Dense pools of size at
  most A retain exact quadratic geometry; explicit A=0 is the dense reference.
- **Residual groups**: pair posteriors cost O(N K^2 L). Maximal-clique enumeration can
  be exponential in A, independently of maximum-cut search. Exact anchor-clique counts
  admit or omit whole components before lifting; factored sums/products avoid unnecessary
  enumeration, and a prime-plan bound only orders work. Compiled traversal unions packed
  original-record memberships, preserving nearest-anchor ties and descriptor multiplicity.
  Deduplication checks every packed word on a hash collision, then counts unique lifted
  sizes before the unchanged 64/128 whole-class rule. Only admitted alternative classes
  become Python groups; core/hierarchy groups join separately outside that budget. This removes
  base24's A256 limitation of constructing every admitted clique as a Python group before
  lifted size screening. Packed storage still scales with admitted clique count C times
  ceil(R/64), plus exact hash storage; traversal/lifting remains graph-dependent. The
  output budget does not bound all pre-budget work or prove polynomial clique complexity.
- **Search**: fixed-K work per expansion is roughly O(N L K^3) for O(K) starts at
  fixed proposal budgets/iteration counts. With a K-scaled expansion backstop this
  gives an O(N L K^4) fitting term, not a bound on total discovery: geometry, clique
  families, tied alternatives and solver work must also be counted. Exact maximum
  cuts are confined to at most 12 rows; larger gauge components use heuristics.

At fixed K, L, A and search/iteration budgets, bounded geometry has nominal O(N log N)
sample dependence but potentially enormous graph-dependent constants. This is not an
empirical whole-stage N^1.2 result. Historical matched chr8 discovery took 14.4 s at
N80 and 1,564.7 s at N320 (580 blocks each); N320 did not finish the whole pipeline
within 2,700 s. Profiling confirmed the pre-budget construction/lifting bottleneck. On
the same Ice Lake host with a 24-core allowance, the adverse block fell from 1,625.9 to
9.8 s and from 54.4 to 2.1 GiB peak RSS, retaining scientific outputs with zero observed
continuous drift. Baseline per-lift timing wrappers add overhead, so ratios are indicative.
All 580 N320 chr8 blocks retained saved scientific decisions with zero observed drift in
76.6 s on 76 Ice Lake cores. The saved 1,564.7 s used 112 AMD EPYC cores and different timer
boundaries; this is a historical comparison, not a matched-hardware speedup. A later
matched 76-core Ice Lake comparison measures N320 chr8 at 1,327.6 -> 84.2 s and N160
at 888.4 -> 48.6 s. Those measurements precede the inline-work and chunk-flush refinements;
final timings and downstream validation are tracked in [performance](../performance.md).

Assignment-only fits use the same bounded shared-emission frontier as full fits and omit H work.
The fractional H implementation traverses sample cost rows contiguously and accumulates
ordinary cost sums with a conservative error interval. An allele decision is accepted
only when the interval is clear of every decision boundary; otherwise the accurate
expansion-sum reference decides. The finite term bound is prepared once per workspace;
an infinite bound selects the reference path for validation. Per-sample terms, founder
order, usage sums, quanta, both schedules and stopping rules are preserved. Its temporary term buffers
use 16NL bytes per concurrently fitted panel. Newly computed dense A arrays feed their immediate
H sweep without a redundant sparse-to-dense rebuild. Unknown-founder carrier unions use compiled
Boolean operations, dosage-subtraction starts use equivalent integer array operations, and singleton
endpoint diagnostics avoid component-label allocation. These exact work-sharing changes preserve
objectives, family semantics and search budgets. Independent frozen-reference fits exercise both
schedules, complete families, cache reuse, chunking and thread counts.

Residual partner membership adds a temporary uint8 matrix of at most NK^2 bytes plus route indices.
The evaluated own-score option's 32 MiB scalar memo covers estimated Python keys, values and table
storage; it is separate from dense A scratch and is not a search budget. Parent-ranked replacement
admission does not use that memo.

Memory per block worker is bounded by entry counts of the fit caches (4,096 fits, 16,384
assignments and transitions), the lazily filled bin-pattern tables, an 8 MiB site tile of the
cavity kernel (at least one site, O(N K^2)), a 32 MiB chunk of the residual pair posteriors and
dense or landmark geometry scratch, plus packed lifted memberships/hash storage reserved
from actual admitted counts, component scratch and admitted-group decoding. Contig evidence
is shared, but kept-site selection and sample canonicalisation/neutralisation make private
block likelihood copies; there is no copy-free per-block guarantee. Completed blocks are
recorded in chunks of up to 64 with the completion-driven elapsed-time flush described
above. An interrupted contig resumes with blocks absent from committed chunks.

Each block task reserves its planned peak memory with the run's memory governor before it starts
(`discovery/stage.py`, `block_task_bytes`): a fixed `discovery.stage.TASK_BASE_BYTES = 536870912` (512 MiB), the
block's evidence slices and the workspace estimate of `discovery/fixed_k/workspace.py` (`search_bytes`: the
evidence tables, the bin-pattern tables at their full 1,024 slots per bin, the entry-bounded caches and the
fixed-point set) for panels of up to `discovery.stage.ESTIMATE_FOUNDERS = 32` rows. Panels and cache keys are
bit-packed, the search archive keeps per K only the panels the search reads (its beam), and duplicate rows are
found with a lexsort over byte rows, never by sorting a structured data type. Block workers return freed heap
pages to the system every 32 blocks, or as soon as their resident set reaches 1.5 GiB.

In the earlier implementation benchmark on the package's K-ladder simulation (N = 320), a block took 4.3, 12.1 and
33.4 CPU seconds at K = 12, 20 and 30 (cavity scoring 73% of it at K = 30), with a worker peak of 0.70 GiB at K = 30
([performance](../performance.md#founder-counts)). These historical timings do not measure the adopted 2.3/2.4 search. On one matched local two-thread TroMau critical block, two unprofiled repetitions of
the exact work-sharing bundle averaged 23.3224 s versus 30.6957 s for its frozen parent-ranking
reference, a 24.02% reduction, with identical scientific and search summaries. This is evidence for
that block only. Independent checks matched every returned fit field and certificate in 108 cases
across one and two threads; neither check establishes whole-stage accuracy or resource acceptance.

On matched full nodes with two threads per worker, the exact speedups reduce the parent-ranking
TroMau sample time from 43.056 to 33.621 s (76 CPUs, IceLake) and K30 from 119.192 to 108.122 s
(112 CPUs, Sapphire). The resulting ratios to 2.2 are 1.848x and 1.938x, with identical outcomes
and selected objectives. These sampled fixed-worker measurements do not establish production
allocation or whole-run performance. Full TroMau subsequently completed; see the versioned
[measurements](../performance.md#completed-integration-measurements). Complete pipeline
reversal and cross-version accuracy attribution are not supplied by those timings. Discovery at 2x 2.2 and whole TroMau at 47:59 are timing guides;
report actual ratios and quality gains for any excess. The historical 62.55-GiB comparison proxy is not a runtime memory throttle; available
node memory and normal headroom govern admission.

## Bounded 2.5 scientific closure

Bounded v2.5 discovery analysis closes D9 with 0/5 released changes on three K30 simulated and two TroMau blocks despite 40/81 changed cut sets (28/81 changed best cut weights); the three simulated blocks retain 16,256 calls, 6 called errors and 1,750 wrong-or-missing cells out of 18,000 truth cells. D9p/D3/central sample order has 0/15 released changes in the fresh controls. Across four retained mean-field cases, 5/53 scored panels fail only in losers, with zero selected or nearest-competitor failures; a stricter solve clears 470 failed held-out/site cells with maximum score drift 3.13e-10 and no class, K, panel, alternative or call change. I6 remains analysis only: the captured TroMau binding class has 50 omitted starts, 44 assignment-masked content classes and 50 distinct full-state signatures, so this capture supports no all-future-context quotient. These are bounded evidence closures, not universal invariance or full-pipeline proofs.

D9 remains heuristic above K=12, and D3 still carries only its two documented schedules.
The closure is a decision against further changes on the bounded evidence, not a proof
that every graph, sample ordering or failed solve is harmless. No fixed-K or cavity solver
change is included in W3. G0 inclusion is unchanged.

## Interpretation and limitations

- The selected K maximises a cavity pseudo-score over a generated neighbourhood. It is not marginal
  evidence for K or an estimate of the ancestral founder count, and its certificate is local.
- A block's rows are a within-block mosaic summary of the samples' haplotypes; they need not
  correspond one-to-one to ancestral founders, and local recombinants or noise rows can appear.
- The released calls are fixed-assignment pseudo-probabilities, not posteriors: the assignments
  come from the same data and the cap bounds each carrier's influence. The supporter count includes
  carriers of either direction, unlike local selection's favourable-carrier rule.
- Unknown copies absorb chromosome copies the panel does not explain; `unknown_mass` reports their
  fraction and is not a probability.
- Above 12 rows gauge rewires use heuristic maximum cuts and may miss the best rewire.
- A cycling fixed-K fit returns its cycle's lowest-NLL state, which need not be a local optimum.
- Values in one tie class remain tied except for stated preferences and the wide-class rule. Complete fitted
  alternatives are retained. Omitted candidate classes describe the computational policy; they do not imply
  biological ambiguity and do not withhold incumbent calls.
- More search can lower a K's score: cavity scores are evaluated over its leading NLL class, and a
  lower-NLL class found later may have worse cavity scores, which can move the selection to a
  smaller K (the broader 2.1 search does this in a few blocks of the simulations; see the 2.1 notes).

## Bounded proposal geometry in 2.3


For K=1 seed pools or residual pools exceeding 256 records, discovery constructs
proposal geometry on at most 256 data-derived landmarks. Exact duplicate descriptors
share an identity; original observation weights are preserved. The initial anchors
are the complete nearest class to the population prototype; subsequent anchors
are complete farthest-from-current-set classes. Distance comparisons use the
existing probability quantum 2^-20, including its chained-class definition.
Names, record order, random seeds and truth do not select members. An over-budget
class is omitted whole and recorded in search_incomplete with cause/count/scope.
If the first class is too large, this proposal branch contributes no groups;
other existing starts remain eligible. Such an omission is a computational limit.

The existing hierarchy and complete maximal-clique family operate on the anchors.
Every original record whose nearest-anchor class intersects a proposed group is
included once in that group's consensus. All tied nearest anchors are retained.
Original responsibilities and per-sample caps are unchanged; several alternative
groups may contain the same observation without multiplying its weight within a
group. Support ranking uses original membership sizes. Pools at or below the
budget take the original exact route. The old density hierarchy is not preserved
for larger pools, so proposals and fitted outcomes can change. All observations
remain in final fitting, held-out predictive scoring, and release.
