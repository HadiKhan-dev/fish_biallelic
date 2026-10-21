# Local selection

Local selection refits every informative discovery block under a normalised diploid copying model
and selects its founder panel by a drop / merge / add search on a regularised likelihood. It runs
three rounds per contig: an initial round from the raw [discovery](discovery.md) panels, then two
context rounds in which the current selection is assembled through assembly level 1 and through
levels 1 and 2, each assembled component's founders are refitted from every covered block's own
reads, and every block is selected again with those refitted founders as candidates. The panels of
the last round are the input of the final [assembly](assembly.md). These are local block
operations, not pedigree or family-phase feedback.

## Inputs and outputs

| | |
| --- | --- |
| input | the contig's `evidence` record in the founder view (samples flagged by the contamination QC are unobserved; see [evidence](evidence.md)), restricted to discovery sites; the discovery records (the latent panel of every informative block); the contig's genetic map; the local-selection configuration (including the copying redraw intensity); the assembly configuration (context assemblies) |
| product | one `LocalPanel` (`model/local_panels.py`) per informative block and round, in block order: positions, discovery sites, released calls `int8[K, L]`, ALT probabilities, favourable carrier support, and the per-sample unknown-copy summary that assembly's completion reads |
| records | `local_selection/<round>/...` for the rounds `initial`, `l1`, `l2`; `identity/local_selection/<round>` |

The stage has no options of its own; its constants are those of `LocalSelectionConfig`
(`local_selection/config.py`), listed in [configuration](../configuration.md). The genetic map
settings `[genetic_map] file` and `[genetic_map] rate_cm_per_mb` set the copying transitions, and
`[execution] batch_queue` lets helper allocations take a round's blocks (timing only).

## Rounds

| round | candidates of a block | context |
| --- | --- | --- |
| `initial` | the raw discovery latent panel; the search starts from it | none: every row is coded as a new haplotype |
| `l1` | the latent panel, the `initial` selection's calls and the proposals refitted from the selection assembled through level 1 | the assembled founders that cover the block |
| `l2` | the latent panel, the `initial` and `l1` selections' calls and the proposals refitted from the `l1` selection assembled through levels 1 and 2 | as above |

`LocalSelectionConfig.context_levels = (1, 2)` sets the assembly levels of the two context rounds.
Each round selects every informative block once; there is no further feedback after `l2`. The
same founder-view likelihoods and observation mask at the same discovery sites enter every round:
assembly contributes candidate sequences, never extra observations.

## Copying model

For a panel H of K binary founder rows over the block's L discovery sites, each of a sample's two
homologues copies a founder row or the *unknown* state (`local_selection/path_model.py`). The prior
state probability is pi = ((1 - u) f, u) with fitted founder frequencies f and the fixed unknown
mass u = `LocalSelectionConfig.model.unknown_mass = 0.01`. An unknown copy draws its alleles
independently as Bernoulli(1/2) at every site; it is not one shared unknown founder. At the first
site each homologue's state is drawn from pi; between adjacent discovery sites each homologue
redraws its state from pi with probability r = 1 - exp(-g dM), so the per-homologue transition is
T = (1 - r) I + r 1 pi^T, where dM is the genetic distance in Morgans between the two sites (from
the contig's genetic map, or the fallback rate without one) and
g = `LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0` is the redraw intensity per Morgan.
The same intensity controls the block path HMM and the context-refit HMM. A redraw can select the
current state again, so g is not the observed founder-switch density: at stationary state
frequencies pi, the expected state-change intensity is g times (1 - sum_i pi_i^2). It is not
a generation label and is not fitted from painting counts. Assembly's independent
`AssemblyConfig.gap_horizon_scale = 3` limits linker and beam proposals; it does not set these
copying transitions. Both settings are internal constants, with independent roles, and neither
is a public TOML setting.

A bounded chr19 sensitivity study on std, F3 and TroMau retains copying intensity 3.0 and assembly
gap scale 3. Copying intensities 1 and 6 both worsen TroMau consistency against the G0 pseudo-truth;
F3 gains in genotype recovery at intensity 1 trade against worse founder recovery. The tested
assembly scales 1 and 6 offer no consistent gain. These painting-stage comparisons support keeping
the defaults, not treating 3 as optimal or interpreting pseudo-truth as known biological truth.

At an observed site, a pair of copy states emits the sample's normalised genotype likelihood of its
dosage: GL[H_a + H_b] for two founders, (GL[H_a] + GL[H_a + 1]) / 2 for a founder and an unknown
copy, and GL[0]/4 + GL[1]/2 + GL[2]/4 for two unknown copies. Unobserved cells emit 1.

The likelihood sums over all diploid copying paths (ordered state pairs) with their normalisation,
rather than choosing one fixed founder pair per sample. The two homologues share priors and
unphased emissions, so the forward messages stay symmetric: trajectories store only the upper
triangle while every ordered state's mass is counted. This symmetry is an exact property of the
model, not frequency tying or state pruning.

## Objective

A fitted (H, f) is scored by its log likelihood plus a coding regularizer
(`local_selection/objective.py`):

    J(H, f) = log P(GL | H, f) - log C(2^L, U) - c_claimed - log K - log(K + 1) + (1/K) sum_a log(K f_a)

- **Dictionary.** log C(2^L, U) is the code length of U rows coded as new arbitrary binary
  haplotypes. In the initial round, and in blocks no assembled context covers, U = K. In a covered
  block of a context round, each refitted context founder claims one of its nearest panel rows on
  the founder's called sites. The claims are those of the shortest description of the panel (release
  2.1, a declared model change): a maximum-cardinality, then minimum-edit-cost matching of founders
  to their nearest rows, so the code is a function of the row set whatever the row order (2.0 let a
  founder equidistant to several rows claim the lowest of them, a greedy rule applied to a
  trajectory-dependent row order); a claimed row is a local parameter of a founder
  whose existence the multi-block assembly decided, coded by reference to that founder plus its
  edits, log C(n_f, d) for d differing sites among the founder's n_f called sites (the smallest
  over the row's claimers, and never more than one more free row would cost). c_claimed sums these
  codes and U counts the unclaimed rows, so a claimed row equal to a proposal costs nothing.
- **K prior.** -log K - log(K + 1), the telescoping prior also used by discovery.
- **Frequency shrinkage.** (1/K) sum_a log(K f_a) is zero at uniform frequencies and negative
  otherwise; it amounts to one extra count spread uniformly over the founders.

This is a regularised best-panel objective, not marginal evidence for K or an assumed ancestral
founder count. Rows beyond the context's founders (local recombinants, noise rows) pay the full
dictionary code.

## Fixed-K fit

A fit (`local_selection/fit.py`) starts from a panel (uniform frequencies unless given) and
alternates generalised-EM updates, each computed from the forward-backward statistics of the
current (H, f):

1. a frequency step, f' = (c + 1/K) / (sum c + 1) with c the expected initial and redraw-destination
   copies of each founder, kept when the objective does not drop by more than the tolerance;
2. a flip step: the founder with the largest summed positive expected complete-emission flip gains
   flips all its sites with a gain above the tolerance; at a stationary point of these gains, the
   single exact conditional one-bit flip with the largest gain is taken instead.

The fit converges at a conditional fixed point: no frequency gain and no exact one-bit flip gain
above `LocalSelectionConfig.search.tolerance = 1e-6` nats. A flip that would make two rows equal
stops the fit, because changing K belongs to the outer search. Founders, or single exact flips,
whose gains lie within a relative 2^-20 of the largest (`local_selection.fit.FLIP_TIE = 9.5367431640625e-07`)
are tied. The data decide between them by the exact objective after each flip. If tied moves still
lead to different panels, the fit stops there (keep-current; termination `tied_flips` or
`tied_exact_flips`), so neither row order nor the last bits of the gains steer it. A fit is returned
with its rows in lexicographic order (frequencies and per-row posteriors permuted with them): the
likelihood and the objective do not depend on the row order, and two fits ending at the same
founders are the same panel to every comparison, tie-break and record downstream. The update backstop is
20 + 4K + 2L updates (`LocalSelectionConfig.search.update_backstop_base = 20`,
`LocalSelectionConfig.search.update_backstop_per_founder = 4`,
`LocalSelectionConfig.search.update_backstop_per_site = 2`), a reported limit. At fixed K the
dictionary term is constant, so a fit maximises the likelihood plus the frequency shrinkage; the
context code enters where fits of different panels are compared. A fit reports, per founder
allele, the exact one-bit conditional probability of ALT (every other bit and pi fixed, equal
prior odds) and the expected copies per site.

## Panel search

The search (`local_selection/search.py`) starts by fitting the round's start panel. Every decision
between objectives in the search and between rounds uses anchored tie classes (`maths/ties.py`) of
the decision tolerance `LocalSelectionConfig.search.decision_tolerance = 0.0009765625` nats
(2^-10): values whose consecutive gaps are at most the tolerance are tied, and a tie is decided by
content (fewer founders, then the canonical panel bytes, then the order in which the candidates
were generated), never by the last bits of the objectives. The tolerance lies far above EM
convergence residuals and the motion of 1e-9 relative GL perturbations, and far below the
differences between distinct drop, merge and add outcomes. In a context round, when the start's fit
reaches the previous round's selected panel of the block (the *incumbent*), the search continues
from the incumbent's fit; when the incumbent scores better than the start's fit by more than the
tolerance, the search explores from the incumbent instead (refitted only when that improves it by
more than the tolerance), so it does not climb back to it. Each search round then:

- proposes every **drop** (one row deleted, frequencies renormalised), every **merge** of rows
  i < j (row j deleted, row i set to the occupancy-weighted consensus of the two rows' allele
  probabilities, its frequency f_i + f_j; where that probability is 1/2 within
  `local_selection.search.CONSENSUS_TIE = 4.656612873077393e-10` (2^-31, placed by the valley census
  between the motion of 1e-9 GL perturbations and the closest genuine values), row i's allele is
  kept) and every **add** of
  a candidate-bank row not in the panel (frequency 1/(K + 1), the others scaled by K/(K + 1));
  starts with the same panel and frequencies equal up to rounding are proposed once;
- screens drops and adds by their exact forward log likelihood plus the coded regularizer, and
  merges by an EM lower bound: the merged panel is the drop panel of either row with the other row
  replaced by the consensus, so the drop panel's posterior gives LL(drop) plus the replaced row's
  complete-data flip gains at the changed sites plus the expected-copy frequency term, and the
  larger of the two bounds ranks the merge. For every row k, all merges with tied nearest rows
  and all merges not involving k with tied nearest consensus rows are scored exactly as well.
  Hamming distances are integers: the complete minimum class is retained and the union is
  deduplicated. The fixed-posterior bound credits no re-routing of carriers, so selecting one
  tied distance by row or proposal order could hide an improving merge. Without distance ties
  this adds at most 2K forwards; with ties it can include every proposed merge. Likelihoods,
  objective, refit caps and stopping rules are unchanged;
- refits whole tie classes of each kind's screened scores, in order, until at least
  `LocalSelectionConfig.search.refits_per_kind = 8` proposals; a class that would take the kind past
  `LocalSelectionConfig.search.refits_per_kind_cap = 16` is left out whole and counted as an overflow (the block's
  summary records `cut_overflow`), so no member of a tie class is preferred to another. Each is refitted to a
  conditional fixed point, and the search
  moves to the best refit (the best tie class of the objective; inside it fewer founders, then the
  panel bytes, then the refit order) when it improves the selection objective by more than the
  decision tolerance.

The search stops when no refit improves; the round backstop 10 + 2K
(`LocalSelectionConfig.search.round_backstop_base = 10`,
`LocalSelectionConfig.search.round_backstop_per_founder = 2`) is a reported limit. After the
search, the incumbent is kept unless the search result beats its selection objective, both under
the current round's context code, by more than the decision tolerance (hysteresis): a context
round replaces a block's panel only for a real improvement, so a change elsewhere in the contig
does not move a block between equally good optima. A block whose search inputs (start, candidate
bank, likelihoods, map distances, settings and context founders) equal the incumbent's reuses the
incumbent's search. Fits, scores and merge bounds are deterministic functions of their
inputs; they are memoised within a block and carried to the block's next round.

## Candidate bank

In the initial round the start is the raw latent panel and the bank its rows. In a context round,
the block's candidate panels are first completed: a partial row (-1 cells) takes the alleles of its
nearest latent rows on its called sites (every tied completion enters the bank; called alleles are
never overwritten). When equally near latent rows give different completions, the panel becomes one
source per combination of its tied rows' completions, so no donor is preferred. A panel with more
than `LocalSelectionConfig.bank.completion_variant_cap = 8` combinations keeps only its rows with a
unique completion as its source; this is counted as an overflow. Every source (the latent panel, each earlier selection, the newest proposals)
is refitted with discovery's fixed-K model and its endpoints are cavity-scored
(`local_selection/bank.py`). Every synchronized and refitted endpoint uses discovery's fractional
A step: each sample contributes uniformly over all tied states in its winning unknown-copy
category. Those same memberships feed the founder field, leave-one-out cavity scoring and the
raw-discovery release of each start. The compact meet pair is only a summary and does not discard
a tied sample's founder evidence. Samples without observed sites are removed from the cavity
likelihoods and their memberships together. Every bank path evaluates both consistent usage-tie
schedules when a positive-usage tie occurs, sharing exact A work and keeping every distinct endpoint
and lower-K collapse descendant in its source family. It also shares an identical H transition when
tied updates provably commute or the sweep changes no allele; both trajectories and their separate
fit certificates remain required (see [discovery](discovery.md#implementation-and-cost)).
The fractional allocation and schedule
families leave the BIC, switching-Viterbi NLL and subsequent path objective unchanged. The separate
[D12 cavity policy](discovery.md#cavity-score) deliberately changes tied-endpoint predictions to a
mixture with equal endpoint-class weights and mean predictive likelihood within each class before
the log; source cavity comparisons use that same functional.

The union bank and the drop, add and replace neighbourhoods of its best panels are then searched
under discovery's switch-penalised binned-Viterbi NLL in a BIC form,
K x `LocalSelectionConfig.bank.complexity_scale = 0.5` x (L / 200) x N_active + 2 NLL (lower is
better; N_active the samples with depth), until it improves by at most the block's `improvement`
tie quantum (2^-14 nats). The block's discovery tie quanta define the bank's tie sets, which are
kept whole (no member is preferred by its key). The selected panels are the best tie class of the
BIC score (kind `nll`), inside it the best class of the NLL, then fewer founders (a class wider
than the width guard keeps only the panels exactly at its best value). Every member's
neighbourhood is searched and every member's rows join the bank. The candidate starts are each
source's cavity-best tie set, chosen the same way from its cavity scores.

Score-cap admission preserves complete source families. Families sharing any endpoint are joined
into atomic components and ranked by their best actual endpoint NLL using the existing `nll`
quantum; complete tied classes are admitted together. The cap counts unique new endpoints. If the
next complete class cannot fit, it and all lower-ranked classes are omitted, `budget_bound` is
reported, and already valid incumbents remain. There is no doubled-cap allowance in this bank
policy. If no initial family fits, the existing original latent incumbent and its rows remain;
no BIC winner or new calls are invented. The separate pre-fit start backstop still limits novel
starts in enumeration order and remains a convention to test; complete endpoints of an admitted
start are never cut to an index prefix. Reported backstops, with K the
current panel's founder count: at most 512 + 64K novel starts per call
(`LocalSelectionConfig.bank.start_backstop_base = 512`,
`LocalSelectionConfig.bank.start_backstop_per_founder = 64`), 1024 + 128K admitted panels
(`LocalSelectionConfig.bank.score_backstop_base = 1024`,
`LocalSelectionConfig.bank.score_backstop_per_founder = 128`) and 8 + ceil(K/2) rounds
(`LocalSelectionConfig.bank.round_backstop_base = 8`).

The candidate starts are the cavity-best tie sets of every source, each member released with the
raw discovery rule and the block's tie quanta (released alleles, latent alleles where withheld). An
incomplete discovery search with an incumbent follows this same path. The exceptional block with
no incumbent because every initial class was omitted contributes no local panel; its unresolved
status and omissions remain recorded in discovery, and no calls are supplied to assembly. Each
candidate start is fitted. The best tie class of their selection objectives (fewer founders inside
it) is searched from every distinct fitted panel. A lower-scoring initial fit is not searched merely
because it belongs to a different source; a better initial objective does not guarantee a better
completed search. The incumbent can still supply the starting fit under the score/reached-panel
rules above. Completed outcomes are compared by the selection objective's tie classes, fewer
founders inside the best. Distinct panels still tied (`tied_alternatives`)
are released as their consensus. A founder row that every alternative contains keeps the calls all
alternatives' releases agree on, their mean ALT probability and their smallest support. The other
rows are withheld (`withheld_by_ties`). Unknown slots are the largest over the alternatives. The
next round's incumbent is the alternative with the smallest canonical key, a convention the released
panel does not depend on. In the initial round, discovery's NLL-tied alternative panels of the block
(`DiscoveredBlock.tied_panels`) join the latent panel as candidate starts, and in context rounds as
sources. The rows the add move may use are the raw latent rows, the candidate starts,
every member of the BIC tie set and the incumbent's rows. Neither the BIC nor the cavity score
replaces the path objective; they only supply starts and rows. Completed latent alleles are not
released as calls by themselves.

## Context rounds

**Context assembly.** Round `l1` assembles the `initial` panels through level 1 and round `l2`
assembles the `l1` panels through levels 1 and 2, with assembly's role CONTEXT: the same assembly
and completion as the final assembly, without founder refinement (see [assembly](assembly.md)).

**Coverage.** A context component covers a block when its sites are exactly the sites of whole
consecutive blocks. Every covering component is refitted, whatever its founder count; a component
that covers blocks only partially is skipped and its blocks keep their current selection as their
proposal, with every row coded free. A completed assembly round therefore does not imply that
every block receives a new proposal.

**Refit** (`local_selection/context_refit.py`). The component's distinct pre-fill rows are its
founders (-1 = uncalled), taken in lexicographic order, so the refit does not depend on the
component's founder order. Per sample, a bin-level diploid HMM has one bin per covered block and as
states the ordered pairs of the component's K founders and a background state, whose alleles are
Bernoulli(site background frequency) per copy, the background frequency at a site being
(1 + called ALT) / (2 + called founders). Transitions redraw toward pi = ((1 - u)/K, ..., u) at
probability 1 - exp(-g D) over the distance D between block centres (Morgans with a genetic map file,
else bp times the fallback rate), with u the unknown mass above and
g = `LocalSelectionConfig.model.redraw_rate_per_morgan = 3.0`, the same redraw intensity used
by the block path HMM. A pair's bin emission sums, over the bin's observed discovery sites at which some
founder is called, the logarithm of a robust genotype likelihood (1 - m) GL(pair) + m/3, with
m = `LocalSelectionConfig.context_refit.emission_mix = 0.01`.
The *cavity* weights of block b, its pair posterior from a forward-backward pass that excludes
block b's own emission, weight each sample in the shared founder-allele site posterior of the
assembly library, which refits every founder allele at each site of block b from that block's own
reads.

The site posterior is exact for at most
`assembly.founder_alleles.site_posterior.MAX_EXACT_BITS = 10` founders (all 2^K configurations
enumerated). Above that it uses the polynomial uncertain core: a posterior mode by loopy belief
propagation and single/pair-flip ascent from two starts; founders whose single flip and every pair
flip cost more than `assembly.founder_alleles.site_posterior.SITE_CERTAIN_NATS = 20.0` nats stay at
the mode; the remaining core is enumerated exactly when it has at most ten founders, otherwise over
a Hamming ball of at most `assembly.founder_alleles.site_posterior.SITE_BALL_CONFIGURATIONS = 1024`
configurations (radius at least `assembly.founder_alleles.site_posterior.SITE_BALL_MINIMUM_RADIUS = 3`).
There is no founder-count cap. An allele is released at posterior probability
`LocalSelectionConfig.context_refit.posterior_threshold = 0.99` with carrier support of at least
`LocalSelectionConfig.context_refit.min_support = 2.0`, by count (samples more likely than not to
carry the founder) or by carrier mass; nothing is released at a site whose two ascent starts end in
different comparable optima (multimodal abstention), or where the ball's shell guard withholds it.
The refit's founder prior is uniform, so a carrier probability can be exactly 1/2 by symmetry and a
carrier mass exactly at the threshold (a sum of halves): within
`assembly.founder_alleles.site_posterior.SITE_HALF_BAND = 1.4901161193847656e-08` of 1/2 a sample is
not counted, and a mass within that band of the threshold does not release (a fixed conservative
rule at a mathematical tie). An ascent flip, a ball recentring and the complement start's optimum
must raise the log posterior by more than
`assembly.founder_alleles.site_posterior.SITE_IMPROVEMENT = 7.62939453125e-06` nats. The [assembly](assembly.md) page describes this posterior in
full. A withheld allele keeps the component's published (completion-filled) call when it has one.

**Proposals.** A covered block's proposal is the set of distinct local rows of the refitted
founders, in lexicographic order; they are its candidate panel and the context founders of the
dictionary code. Proposals are recorded per round.

## Release rule

The selected panel's alleles are released from one more E-step over per-sample posteriors
(`local_selection/release.py`). For founder k at site l:

- the one-bit conditional probability of the panel's allele is 1 / (1 + exp(G_kl)), where G_kl is
  the log marginal likelihood ratio of flipping that allele with every other bit and pi fixed
  (summed over samples, equal prior odds);
- the *favourable* samples are those whose own exact flip gain is below
  -`LocalSelectionConfig.release.favourable_margin = 1.1920928955078125e-07` nats (the `cell` tie
  quantum 2^-23), i.e. whose evidence favours the allele; a symmetric-evidence cell (one REF and one
  ALT read: GL(0/0) = GL(1/1)) gives a homozygous carrier a gain of exactly zero, inside this band,
  so it is never favourable; the carrier probability is the posterior probability that a homologue
  copies founder k at the site (a homozygous state counted once);
- the allele is released when its probability reaches
  `LocalSelectionConfig.release.min_call_probability = 0.85` and the favourable samples include
  `LocalSelectionConfig.release.min_supporters = 2.0` carriers, by count (samples with carrier
  probability above 1/2) or by expected carrier mass. The mass alone sits just below two for a
  founder with exactly two near-certain carriers; the count alone would withhold diffusely
  supported sites; so either suffices.

Released alleles are the panel's own; others are -1, as are all sites that are not discovery
sites. The probability is a conditional one-bit quantity, not a full allele posterior or a
calibrated probability of correctness. The `LocalPanel` stores the fit's ALT probabilities and the
favourable carrier counts.

**Unknown-copy summary.** For each sample with depth, the largest MAP number (0, 1 or 2) of copies
drawn from the unknown state over its observed discovery sites (2 for a sample without depth), and
the block's `unknown_mass`, those copies over the 2 copies of every sample with depth. A site's
MAP count must win by more than `LocalSelectionConfig.release.unknown_count_band = 9.5367431640625e-07`
(2^-20, the `probability` tie quantum) times the largest count mass, else the smaller count is
taken: completion's veto is all-or-nothing, and count masses can tie exactly. Assembly's
completion reads them as a spatial MAP summary; it is not a probability of any unknown state and
does not invent a constant founder pair for a sample.

## Conventions and stated priors

Where tied alternatives remain after every data comparison, the rules below decide. The *conventions* carry no
biological meaning and are reversed when `HAPLOTYPES_TIE_REVERSE=1`
(`maths.ties.reverse_conventions`; the release 2.2 acceptance test). A released result that changes under the
reversal was chosen by a convention. The conventions are:

- the novel starts kept when the bank's start backstop binds (the first in enumeration order);
- the fit kept per bank key (the first admitted);
- the internal representative of distinct panels left tied after the multi-start search (the smallest canonical
  key; the released panel is their consensus);
- the content order inside tie classes (canonical keys, panel bytes, generation and trial order; reversed centrally
  in `maths.ties.tie_classes`);
- the merge consensus allele at a probability of exactly 1/2 (row i's, the lexicographically smaller row);
- in the site posterior: the founder order of tied ascent moves and of tied ball recentrings, and the REF start of
  an exactly zero BP marginal.

In the context refit, duplicate founder rows share the consensus of their released templates (-1 where called
templates differ). Duplicate local rows carry their summed support and support-weighted ALT posterior. A bin's background-copy
count is the smallest count within a relative 2^-20 of the largest state-pair weight
(`local_selection.context_refit.BACKGROUND_COUNT_BAND = 9.5367431640625e-07`). These are data decisions, not
conventions.

The *stated priors* are model preferences and are not reversed:

- fewer founders inside a tie class;
- keep-current inside a tie class (the climb stops unless a gain exceeds the decision tolerance; the previous round's
  panel is kept unless beaten by more than it; a start that reaches the incumbent's panel continues from the
  incumbent);
- the conservative side of an exact threshold (the favourable band, the smaller unknown-copy count, the half bands
  of the context refit and of the site posterior);
- fewest flips first in the site posterior's ascent and ball;
- a GEM fit stops where tied moves lead to different panels (keep-current);
- the BP basin over its complement at equal optima, which are flagged multimodal and not released.

Proposal starts whose frequencies agree to within `local_selection.search.PROPOSAL_FREQUENCY_TIE = 1e-12` are one
start, not alternatives. Refits that reach the winning panel from different proposals are one answer, differing
only by EM residuals; the refit of the proposal with the smallest (kind, proposal panel) label represents it.
Canonical row orders and these labels are not conventions.

### Observed effects of the start cap and inner winner order

A census of 318,048 block-rounds across eleven datasets finds nine start-cap events in six
block-rounds, all on TroMau or K30. The cap is the 512 + 64K novel starts per bank call stated above.
There are 73 same-K distinct-winner ties among 311,074 inner winner decisions; six advance the
search and 67 are terminal. Both conventions therefore bind internally.

Replaying 223 flagged block-rounds with the global convention reversal and with reversal isolated
to the inner winner's content key changes no released local calls, K, missingness, favourable
support, unknown-copy summaries or local alternative/incumbent decisions. Native propagation
checks on K30 m1 (40 blocks) and TroMau chr7 (834 blocks) retain these results through all three
rounds. The start-cap check uses the global hook, not an isolated start-cap-only intervention.

K30 retains one changed latent allele that is uncalled in both releases, with maximum ALT-probability
drift 6.7e-6. TroMau's initial drift disappears after the first context round. These are measured
search-trajectory effects, not a claim that all numerical fields are identical. The evidence
does not justify a remedy for these two conventions and does not establish invariance on untested data.

The global hook also reverses assembly conventions. It changes one TroMau L2 context-assembly
`release_alternatives` record and some K30 context-proposal probabilities; neither effect occurs
under winner-only reversal, and neither changes the released local calls or decisions in these
checks. The [assembly follow-up](assembly.md#context-assembly-convention-follow-up) is separate
from the two local-selection conventions. Final assembly, painting, pedigree, phase and exported
outputs are outside the propagation check.

## Records

| key (under `local_selection/<round>/`) | content |
| --- | --- |
| `blocks/<n:06d>` | chunk n, in write order, of up to 64 completed blocks: their `LocalPanel`s, fits and search outcomes, releases and summaries |
| `memos/<n:06d>` | rounds `initial` and `l1`: the same blocks' likelihood memos for the next round (written before the chunk) |
| `proposals` | rounds `l1` and `l2`: the context proposals and the refit summary |
| `selected` | round completion |
| `assembly/...` | rounds `l1` and `l2`: the context assembly's own records (joined batches are shared by all assemblies of a contig) |

The identity of the `initial` round binds the local-selection and discovery configurations, the
copying redraw intensity, the genetic map and the evidence and discovery record identities, with the
code of local selection and discovery only, so assembly changes leave the initial fits valid. The
context rounds add the assembly configuration and the previous round's identity. Readers:
`local_selection.records.load_panels` (the final assembly reads round `l2`), `load_selections`,
`panels_identity`, `round_complete`, `complete`, `progress` and `founder_panels` (per-round calls,
for evaluation).

## Implementation and cost

| module | role |
| --- | --- |
| `local_selection/stage.py`, `local_selection/rounds.py` | the three rounds; per round one pool task per block, or cross-node batches |
| `local_selection/path_model.py` | forward-backward of the copying model: lane kernels for scores and fit statistics, the per-sample release E-step |
| `local_selection/fit.py`, `local_selection/objective.py`, `local_selection/search.py` | GEM fit, objective, drop / merge / add search and memos |
| `local_selection/bank.py` | candidate bank (discovery's fixed-K fit and cavity score) |
| `local_selection/context_refit.py` | context refit on `assembly/founder_alleles/site_posterior.py` |
| `local_selection/release.py`, `local_selection/records.py` | release, `LocalPanel`, records |

Blocks are independent within a round. A round runs one task per block in a process pool over
shared evidence arrays (or as cross-node batches of consecutive blocks), and the context refit runs
one task per component. The path kernels process samples in lanes of up to 64 side by side. For
blocks with more than 64 samples, lanes narrow to 32 or 16 when all resulting lane blocks fit the
held thread budget. Each kernel caps its active threads at the useful lane count and restores the
caller's thread mask afterwards, retaining the CPU claim throughout. Uncached evaluations refresh
the layout when the budget changes. Sample results are combined in the existing order; released
scientific outputs and decisions are unchanged, with continuous drift reported in the
[performance checks](../performance.md#local-selection-measurements-for-26).

Cost in the founder count K, samples N and discovery sites L of one block (S = K + 1 copy states):

- **One forward or backward pass** is O(N L K^2): the redraw transition is separable, so each site
  costs O(S^2) per sample. A backward pass computes only the statistics its caller needs: expected
  copies, then complete-emission flip gains and occupancy, then the exact one-bit gains, which are
  computed only at stationary points and for the reported probabilities.
- **A GEM update** is a few passes, O(N L K^2); a fit takes at most 20 + 4K + 2L updates.
- **A search round**, without distance ties, has the existing O(N L K^3) screening estimate:
  drop forwards, add forwards, drop-statistic passes, merge lower bounds and at most 2K exact
  merge forwards. Complete distance ties can select every available merge, up to K(K - 1)/2,
  giving O(N L K^4) worst-case screening. Refit admission still uses whole score classes,
  with target 8 and cap 16 per proposal kind.
- **The candidate bank** fits at most 512 + 64K starts per call with discovery's fixed-K model,
  completing up to two usage schedules per start only where a tie is encountered, with shared A
  work and certified identical H transitions (see [discovery](discovery.md#implementation-and-cost)).
- **The release** is one per-sample E-step, O(N L K^2).
- **A context refit** of a component with B blocks and L_c sites costs O(N L_c K^2) for its bin
  emissions and O(N B K^2) for the bin forward-backward. The site posterior costs O(N 2^K) per site
  up to the exact switch point K = 10 and polynomial work above it (O(N K^2) per belief-propagation
  iteration and ascent step, O(N (m^2 + 2^10)) for an exact core of m founders, O(N m^3) for a
  radius-3 ball); nothing in it is exponential in K above ten.

The numbers of search rounds, GEM updates and bank rounds are set by convergence; their reported
backstops grow linearly in K (and in L for GEM updates), so the worst case the backstops allow is
O(N L K^3 (K + L)) per block for the search. Memory per block search is dominated by the trajectory
scratch, at most 64 L (S(S + 1)/2 + S) float64 values per active thread (one lane block each), shared
by the block's sequential fits. The memory estimate retains width 64; layout transitions can briefly
retain old arrays. A context refit holds one (N, B, S, S) array per component.

The path kernels evaluate the likelihood and its statistics in one pass per lane block of
up to `local_selection.path_model.LANE_WIDTH = 64` samples (forward, then backward at once), so a fit keeps the
trajectories of one lane block per thread, not of every sample, in one scratch buffer per block search
(`workspace_bytes`); the release E-step accumulates carrier counts inside the kernel in batches of
`local_selection.path_model.RELEASE_SAMPLE_BATCH = 128` samples. Rows are deduplicated with a lexsort, never by
sorting a structured data type.

Every task reserves its planned peak memory with the run's memory governor before it starts. A block task
(`local_selection/rounds.py`, `block_task_bytes`) plans `local_selection.rounds.TASK_BASE_BYTES = 335544320`
(320 MiB), the block's evidence slices and emission tables, the trajectory scratch for the largest panel the search
may reach (its start panel plus `local_selection.rounds.SEARCH_GROWTH = 8` rows, one lane block per thread), the
release batches, the candidate bank's discovery workspace (context rounds) and its likelihood memo with room to
grow. A context refit (`local_selection/context_refit.py`, `refit_task_bytes`) plans its sites' slices, the
(N, B, S, S) emission array, each thread's (B, S, S) buffers, its outputs and 256 MiB for a worker's first task.
Blocks run longest first (by sites x states^2), so a round's tail holds the small blocks; cross-node bundles carry
the same estimates. The order and the pacing change only when work runs, never its result.

Completed blocks are recorded in chunks of 64, their memos before them; a context round's proposals
are recorded before its selection starts; a round is complete when its `selected` record exists.
An interrupted round resumes with its missing blocks, and a contig whose three rounds are recorded
returns the `l2` panels without decoding evidence or fits.

On the package's K-ladder simulation (N = 320, every block with K distinct founders) the initial round took 15.1,
32.1 and 60.8 CPU seconds per block at K = 12, 20 and 30 (worker peak 0.78 GiB at K = 30), round l1 0.4, 3.1 and
9.2, and a context refit 29, 80 and 217 per component ([performance](../performance.md#founder-counts)).

## Interpretation and limitations

- The selected panel is the best local optimum of a regularised objective reached by the search;
  it is not a posterior over panels, and K is not an estimate of the ancestral founder count.
  Within-block mosaics need not correspond one-to-one with ancestral haplotypes.
- Context rounds can only add candidate starts and change the dictionary code; they never add
  reads, and the incumbent is protected against a worse objective. Rare weak rows can still be
  lost when no candidate or move recovers them.
- Released alleles carry conditional one-bit support plus directional carrier evidence, not full
  allele posteriors. Unsupported alleles stay -1.
- The copying model uses the genetic map and one redraw intensity g for all samples and blocks; a wrong map or g
  changes how readily samples switch founders within a block.
- The unknown state absorbs copies no founder explains; a high `unknown_mass` flags a block whose
  panel misses haplotypes or whose reads are poor.
