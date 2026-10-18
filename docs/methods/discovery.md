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

## Ties and order (release 2.1)

No discrete outcome of discovery may depend on floating-point rounding, on perturbations of the genotype
likelihoods far below any real difference, or on the order of the samples. Every block therefore has its tie quanta
(`maths/ties.py`, `BlockQuanta` from the block's null NLL M, the sum over samples of their genotype-oracle NLL plus
two unknown copies at every observed site, which bounds every panel's NLL): powers of two placed between the rounding
and noise motion of the compared quantities and their first genuine differences. Kind `nll` (at least
max(2^-17, 2^-30 M) nats) orders NLLs and cavity scores, kind `sample` (max(2^-23, 2^-36 M)) per-sample sums (A step,
data-birth excess), `h_step` (2^-18) the H step's allele sums, `improvement` (2^-14) the search's improvement tests,
`cell` (2^-23) per-cell release contributions and `probability` (2^-20) probabilities and masses. Values are grouped
into anchored tie classes (single-linkage chains of gaps at most the quantum over the exactly sorted values); every
order is (class, then exact secondary keys, then an exact content key), a winner is the content-smallest member of the
best class, and a class wider than four quanta is ordered by value inside (width guard). A band with a fixed rule is
used only where a value can sit exactly on a threshold mathematically (an exact zero, exactly 1/2, a posterior split
exactly in half); continuous thresholds are compared as they are. The samples of a block are searched in the run's
canonical order (by sample name, `run_sample_order`; the evidence stage provides it), so a reordering of the input
changes nothing. Decisions taken inside a tie, width-guard and boundary-guard events and band decisions are counted in
the block's `SearchSummary.tie_events`.

The same evidence gives the same result on every CPU. Discovery evaluates log and exp with the C library's scalar
functions (Numba kernels), not with NumPy's float64 ufuncs, whose SIMD code paths NumPy chooses at run time from the
CPU (AVX-512 kernels differ from the scalar functions by an ulp in about 5% (exp) and 0.3% (log) of values); its other
arithmetic is element-wise or summed in a fixed order, and it makes no BLAS calls. HDBSCAN (the K = 1 seeds and the
residual-strand clusters) runs the hdbscan package's precomputed-distance algorithm with the minimum spanning tree's
edges ordered by weight and, among equal weights, by a fixed pseudo-random key of the edge's two record indices
(splitmix64; `residual_births.exact_hdbscan_labels`). The package orders them with NumPy's unstable default sort, whose
order among equal weights depends on the CPU's sort kernel, and equal Hamming distances are the rule, so the clusters
and their order changed with the CPU (release 2.0 and 2.1 before this rule: actm3 chr8 block 204 under OpenBLAS
Haswell kernels with NumPy's AVX-512 paths disabled). The key depends only on the canonical record order; it keeps the
unstable sort's effectively random resolution of equal-weight merges, which the 2.1 quality measurements were made
with. (A stable order resolves every multi-way merge as a chain, so equidistant residual strands fall out of the
growing cluster instead of pairing up: on TroMau 255 blocks changed, K up/down 60/96, net -9.4k nats.)

Ordering equal-weight links by this content key is a deterministic convention, under review for release 2.2: any
fixed order among exactly equal links is a choice the data do not make. Merging equal links simultaneously (the
order-invariant HDBSCAN* hierarchy) lets equidistant strands fall out instead of pairing up and so drops those
candidates (TroMau: K up/down 64/116 and -9.7k nats against this convention's predecessor). The 2.2 plan carries every
tie resolution forward as a candidate and lets the block score decide. Such ties are not confined to Hamming
distances: with min_samples > 1 (the K = 1 seeds) mutual reachability ties links through shared core distances even
when the distances themselves are continuous.

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

- **A step**: each sample's single best state for the whole block, by its summed (uncapped) bin
  emissions: the first state, in the order row pairs (row-major), row with an unknown copy, two
  unknown copies, whose cost is within the block's `sample` quantum of the cheapest;
- **H step**: Gauss-Seidel founder-bit updates for the fixed assignments, founders in decreasing
  order of usage; for founder k at site l the costs of allele 0 and 1 are summed over its carriers
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

## Cavity score

The search compares panels, including panels of different K, by a regularised fixed-assignment
leave-one-sample-out predictive score (`discovery/cavity.py`):

    score(H) = sum_i log sum_s pi_-i(s) prod_l p(GL_il | s, q_l^(-i))  -  log K  -  log(K + 1)  -  log C(2^L, K)

- **State prior.** pi_-i(s) = (n_s - [s = s_i] + alpha beta_s) / (N - 1 + alpha) is the
  Dirichlet-multinomial predictive of the state counts n_s of the panel's assignments with sample i's
  own count removed; alpha = `DiscoveryConfig.cavity.state_concentration = 0.5`. The prior masses
  beta give the row-pair states 1 - omega (a heterozygous pair twice the weight of a homozygous one),
  each row-with-unknown state omega (1 - delta) / K and the two-unknown state omega delta, with
  omega = `DiscoveryConfig.cavity.unknown_prior_mass = 0.02` and
  delta = `DiscoveryConfig.cavity.double_unknown_fraction = 0.25`.
- **Founder alleles.** q_l^(-i) is a mean-field posterior over the K founder alleles at site l
  given every other sample's assignment and genotype likelihoods: a pairwise Markov random field
  with a uniform allele prior, unary terms from homozygous carriers and carriers with an unknown
  copy, and edge terms from heterozygous row pairs, with sample i's own term removed. Each held-out
  solve runs three starts (the panel's alleles, their complement, all one half; founders without
  other carriers stay at one half) for at most `DiscoveryConfig.cavity.mean_field_max_iterations = 100`
  sweeps to tolerance `DiscoveryConfig.cavity.mean_field_tolerance = 1e-10`, and keeps the first
  converged start (in that order) whose ELBO is within the block's `nll` quantum of the best. A
  founder update whose logit is within the same quantum of zero sets its allele probability to
  exactly one half, so an exact symmetry (equal REF and ALT evidence) is not broken by rounding or
  noise. Solves whose three starts all fail are reported.
- **Predictive.** p(GL_il | s, q) is sample i's likelihood averaged over independent founder alleles
  q under state s; an unknown copy carries ALT with probability 1/2.
- **Regularisers.** -log K - log(K + 1) is a telescoping prior on K (the terms 1/(K(K + 1)) sum to
  one), and log C(2^L, K) is the code length of an unordered set of K distinct binary rows of
  length L.

The score is a pseudo-score, not calibrated evidence: the panel and the assignments it holds out
from were themselves selected on the full data.

## Reversible search over K

The search (`discovery/search.py`, moves in `discovery/moves.py`) keeps an archive of every fitted
complete panel (distinct rows), one entry per canonical panel with its first fit (replaced only by a
fit lower by more than the `nll` quantum). Per K the archived panels are ordered by tie classes of
NLL, then canonical key; the first is that K's *representative* and is cavity-scored. The selected
panel is the representative with the best score class (ties: smaller K, then canonical order).
There is no K grid and no K cap besides the identifiability ceiling min(2 N_active, 2^L), N_active
being the samples with depth in the block.

**Starts.** The K = 1 starts are pooled rows of the largest HDBSCAN clusters (at most
`DiscoveryConfig.search.data_seed_count = 8`, minimum cluster size
`DiscoveryConfig.search.seed_min_cluster_size = 2`) of the samples with depth, clustered by their
expected genotype agreement (summed in one fixed order, exactly symmetric); a cluster's mean ALT
fraction within the `probability` quantum of 1/2 takes the population ALT fraction. The sample with
the most observed sites (then the best decisiveness class, then canonical order) gives the fallback
dosage row. They are fitted and at most `DiscoveryConfig.search.data_start_beam_width = 6`
distinct fits are kept.

**Expansions.** Each expansion takes the beam of the selected K, its
`DiscoveryConfig.search.beam_width = 6` first archived panels.

- *Ordinary phase.* The first beam panel not yet expanded ordinarily proposes its deaths, gauge
  rewires and data births. If the selected score did not improve by more than the block's
  `improvement` quantum, birth-delete replacements follow.
- *Residual phase.* Once every beam panel was expanded ordinarily, the first one not yet expanded
  in this phase proposes soft-residual births, then replacements, and, if the score still did not
  improve, a K + 2 bridge: the data and soft births of the K + 1 representative that reach K + 2.

**Moves.** Each move proposes start panels; every start is refitted by the fixed-K fit, and both
its assignment-only fit and its complete refit enter the archive (a refit that collapses rows
re-enters as a start at the lower K).

| move | K | proposal |
| --- | --- | --- |
| death | K - 1 | every one-row deletion |
| gauge rewire | K | for a maximum cut of the assignment graph (rows linked by the number of samples assigned to both), flip the sites where both sides are internally constant and differ: per site to the better allele (by more than the `nll` quantum) for the samples within one side, and all such sites at once |
| data birth | K + 1 | the samples with the largest excess NLL over their sitewise genotype oracle (tie classes of kind `sample`; an all-unknown sample's excess is its exact value 2 lambda n_obs), then more observed sites, then canonical order, propose K subtraction seeds clip(dosage - row, 0, 1) and their dosage row; tied samples form groups consumed whole: starts are collected group by group up to the attempt budget max(16, 2(K + 1)) x `data_birth_attempts_factor = 3` and fitted group by group until (K + 1) x `data_birth_children_factor = 3` distinct complete children exist (tests at group boundaries); a group is cut only at `data_birth_tie_budget_factor = 2` x the budget (limit `data_birth_tie_budget`) |
| soft-residual birth | K + 1 | consensus rows of residual strands (below), hardened at 1/2 (within the `probability` quantum of 1/2 the population ALT fraction decides) |
| replacement | K | a K + 1 birth (in tie-class order of NLL) with any one row deleted |

Deaths and gauge rewires together are capped at
`DiscoveryConfig.search.max_proposals_per_expansion = 128` starts, as are soft births;
replacements at `DiscoveryConfig.search.max_replacement_children = 24`. Gauge rewires use exact
maximum cuts up to `DiscoveryConfig.search.exact_cut_max_k = 12` rows (by connected components:
bipartite components by 2-colouring, the others by exhaustive enumeration of the 2^(m-1) cuts of
a component of m rows); above that, degree-ordered local searches from every singleton and from an
alternating start. At most `DiscoveryConfig.search.max_cut_ties = 4` equal maximum cuts are used.

**Soft-residual births** (`discovery/residual_births.py`). Each sample's posterior over unordered
row pairs (uniform prior over the hard pairs) gives every row an expected copy count; each row with
at least `DiscoveryConfig.residual_births.min_soft_responsibility = 0.25` expected copies in a sample
is subtracted from it, leaving a residual strand whose ALT probability at an observed site is
P(GL | subtracted allele + residual allele) normalised over the residual allele. Residual strands
called at `DiscoveryConfig.residual_births.residual_call_probability = 0.8` are clustered by
missing-aware Hamming distance (HDBSCAN, then complete linkage on its noise; cluster diameter at
most `DiscoveryConfig.residual_births.max_cluster_hamming = 0.1` of the sites), over all strands and
within each (subtracted row, dominant partner) route. A cluster's consensus pools its strands'
responsibility-weighted log odds with one capped vote per sample; it is emitted when at least
`DiscoveryConfig.residual_births.min_unique_sample_support = 2` distinct samples and
`DiscoveryConfig.residual_births.min_effective_sample_support = 1.5` effective samples support it,
at least `DiscoveryConfig.residual_births.min_candidate_known_fraction = 0.8` of it is called at
`DiscoveryConfig.residual_births.candidate_call_probability = 0.9`, and it duplicates neither a
panel row nor an earlier emission. A strand's log odds are log(residual ALT mass) - log(residual REF
mass), within the bounds [log 2^-1022, log((1 - 2^-52) / 2^-52)] (about -708 and 36), computed
from the masses rather than from the rounded probability; the dominant partner is the largest
partner mass beyond a relative `probability` quantum (lower row on ties); a record weight or an
effective support within the `probability` quantum of its threshold counts as at the threshold
(symmetric evidence splits posteriors exactly in half). Consensus rows are emitted with more distinct
supporting samples first, then in the order of their hardened rows, then source order (so of two
near-duplicates the better supported one is kept).

**Stopping and limits.** The search stops when the selected K's beam is exhausted in both phases,
or at the expansion backstop. Backstops scale with K_rep, the largest K represented in the archive:
at most 32 + 8 K_rep expansions (`DiscoveryConfig.search.expansion_backstop_base = 32`,
`DiscoveryConfig.search.expansion_backstop_per_founder = 8`) and at most 96 + 12 K_rep cavity scores
(`DiscoveryConfig.search.exact_score_backstop_base = 96`,
`DiscoveryConfig.search.exact_score_backstop_per_founder = 12`). They are reported limits, not
bounds on the number of founders. The stop reason is `generated_neighbourhood_local_optimum`,
`incumbent_neighbourhood_search_limited` (some limit was hit) or `expansion_backstop`. Limits are
`expansion_backstop`, `exact_score_backstop`, the generator truncations `proposal_budget`,
`replacement_proposal_budget`, `soft_residual_proposal_budget`, `data_birth_tie_budget` and
`gauge_max_cut_ties_truncated`, `mean_field_nonconvergence`, and `escalation_backstop`. Heuristic
maximum cuts above K = 12 are reported separately (`heuristic_gauge_cuts`) and are not a limit.

**Switches (off in 2.1).** `DiscoveryConfig.search.escalation` widens the search once for a block the
base search leaves fragile (selected K >= `escalate_min_k = 9`, a cavity margin over the best other K
below `escalate_margin = 10` nats, or a selection inside a tie): beam `escalate_beam_width = 12`,
replacements `escalate_replacement_children = 48`, at most `escalate_expansions_percent = 100` % of
the expansion backstop more expansions. `DiscoveryConfig.search.reversed_convention_search`
(`fragile` or `all`) runs a second search with the reversed sample order and complemented content
keys whose archived panels join the first search's archive; the selection rule is unchanged over the
pooled archive (within K the NLL classes, across K the cavity score), and the score backstop doubles.

**Certification.** A selected panel is certified when it was expanded in both phases without any
limit, every K its moves generated is scored, and no move improved the score: a certificate of a
local optimum within the generated neighbourhood, not of a global one.

## Release rule

The selected panel's hard rows are *latent*: they are complete at the discovery sites and are what
local selection fits from, but they are not calls. Alleles are released per founder k and site l
with the selected assignments held fixed (`discovery/release.py`). Each observed carrier of k
contributes its capped cost difference min(c0, cap) - min(c1, cap), with the H step's costs and
cap (homozygous carriers g = 0 against 2; carriers with a known partner j, H_j against H_j + 1;
carriers with an unknown copy, the row-with-unknown costs). With E_kl the sum of these
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
| `k_values`, `k_log_scores` | represented K values and the cavity score of each K's representative; the selected K maximises it |
| `natural_k_ceiling` | min(2 N_active, 2^L) |
| `expansions`, `data_starts`, `exact_scores`, `exact_score_cache_hits` | work counts |
| `residual_calls`, `residual_rows` | soft-residual birth calls and the novel rows they emitted |
| `limits` | reported limitations, in first-occurrence order |
| `heuristic_gauge_cuts` | some expanded panel had K > 12 and used heuristic maximum cuts |
| `stop_reason`, `certified` | why the search stopped; the local certificate above |
| `tie_quantum` | the block's `nll` quantum |
| `escalated`, `conventions` | the escalation ran; the number of pooled searches |
| `ambiguous`, `tied_alternatives` | the selection was taken inside a tie (another representative in the selected score's class, or another panel in the selected K's first NLL class), and how many |
| `score_margin` | the selected score minus the best other represented K's score |
| `tie_events` | per decision kind: decisions, decisions taken inside a tie, width-guard and boundary-guard events (diagnostic; not an outcome) |

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
fixed-K fits of many starts advance together as a frontier over a memoised deterministic map
(equal states are evaluated once) and use up to `discovery.fixed_k.fit.FIT_MAX_THREADS = 16`
threads, claimed from finished workers at phase boundaries; the cavity kernel runs held-out samples
in parallel. Every sum over samples or sites runs in a fixed order, so results do not depend on
thread counts, batch composition or the order in which blocks complete.

Cost in the founder count K, samples N and discovery sites L of one block:

- **Fixed-K fit**: the A step evaluates (K + 1)(K + 2)/2 states over L/10 bins per sample,
  O(N K^2 L); the H step is O(N L + K L); at most 50 iterations.
- **Cavity score of one panel**: per site and held-out sample, mean-field sweeps over K founders
  and at most K(K - 1)/2 active edges, and a predictive over O(K^2) states: O(N L K^2) per sweep,
  with exact reuse of equal held-out cases.
- **Soft-residual births**: pair posteriors O(N K^2 L) and a distance matrix quadratic in the number
  of residual strands (at most N K).
- **Search**: an expansion fits O(K) starts (K deaths, at most 2 x 3 x max(16, 2(K + 1)) data-birth
  starts, bounded gauge, soft and replacement starts), O(N L K^3) of fixed-K work. The number of expansions
  is set by the local optimum, bounded only by the backstop 32 + 8 K_rep, so the worst case allowed
  by the backstops is O(N L K^4) per block. The only exponential step, exact maximum cuts, is
  confined to components of at most 12 rows (at most 2^11 cuts).

Memory per block worker is bounded by entry counts of the fit caches (4,096 fits, 16,384
assignments and transitions), the lazily filled bin-pattern tables, an 8 MiB site tile of the
cavity kernel (at least one site, O(N K^2)), a 32 MiB chunk of the residual pair posteriors and
the residual distance matrix. Block tasks read views of the contig's evidence; no per-block
likelihood copy is made. Completed blocks are recorded in chunks of 64, so an interrupted contig
resumes with its missing blocks only.

Each block task reserves its planned peak memory with the run's memory governor before it starts
(`discovery/stage.py`, `block_task_bytes`): a fixed `discovery.stage.TASK_BASE_BYTES = 536870912` (512 MiB), the
block's evidence slices and the workspace estimate of `discovery/fixed_k/workspace.py` (`search_bytes`: the
evidence tables, the bin-pattern tables at their full 1,024 slots per bin, the entry-bounded caches and the
fixed-point set) for panels of up to `discovery.stage.ESTIMATE_FOUNDERS = 32` rows. Panels and cache keys are
bit-packed, the search archive keeps per K only the panels the search reads (its beam), and duplicate rows are
found with a lexsort over byte rows, never by sorting a structured data type. Block workers return freed heap
pages to the system every 32 blocks, or as soon as their resident set reaches 1.5 GiB.

On the package's K-ladder simulation (N = 320, every block with K distinct founders) a block took 4.3, 12.1 and
33.4 CPU seconds at K = 12, 20 and 30 (cavity scoring 73% of it at K = 30), with a worker peak of 0.70 GiB at K = 30
([performance](../performance.md#founder-counts)).

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
- Values closer than a tie quantum are treated as equal and the canonical content key decides among
  them (fewer founders first in the selection); `ambiguous` flags a selection taken inside such a tie.
  The search stays path-dependent: a different tie convention can reach a different local optimum
  (the optional reversed-convention search explores the alternative).
- More search can lower a K's score: the representative is the lowest-NLL panel of its K, and a
  lower-NLL panel found later may have a worse cavity score, which can move the selection to a
  smaller K (the broader 2.1 search does this in a few blocks of the simulations; see the 2.1 notes).
