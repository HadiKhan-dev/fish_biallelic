# Ties and deterministic conventions

A tie is a decision whose alternatives the data do not separate. Either they score exactly the same (for example
two panels that differ only at alleles no sample observes), or they differ by less than the rounding noise of the
compared sums. Release 2.1 decides every tie by a fixed, documented convention. The same input therefore gives the
same output whatever the sample order in the input file, the thread count, the CPU instruction set, or a 1e-9
relative perturbation of the genotype likelihoods. The convention picks one of several equally supported
alternatives. That choice is not evidence for the alternative it picks.

**Status.** These are the release 2.1 conventions. They are deterministic and recorded, but they are **under review
for release 2.2**. Some may be replaced by rules that keep the tied alternatives, or that abstain. The run counts
every decision taken inside a tie (see [What a run records](#what-a-run-records)), so the effect of a convention can
be checked on any dataset.

## Common machinery

`maths/ties.py` gives every stage one way to say "these values are tied" and one way to break the tie.

- **Tie quanta.** Each kind of compared quantity has a quantum, a power of two. It is placed far above the rounding of
  the compared sums and far below the first genuine differences, which were measured by a valley census of real and
  simulated runs. The kinds are block NLLs and cavity scores (at least 2^-17, scaled with the block's null NLL),
  H-step allele costs (2^-18), "improved" tests (2^-14), per-sample sums (at least 2^-23), per-cell evidence (2^-23)
  and probabilities (2^-20). Stages with their own scale use a quantum of the same construction, for example the
  local-selection decision tolerance and the refinement quantum max(2^-14, 2^-35 |F|).
- **Tie classes.** Values are sorted exactly. Consecutive values whose gap is at most the quantum chain into one
  class, so class membership is transitive. A decision takes the first item in the order (class, secondary exact
  keys, content key). The winner's class is anchored at the best value.
  - **Width guard.** A class wider than four quanta is reported and ordered by value instead of content, so genuinely
    different values are never reordered by content.
  - **Boundary guard.** A gap close to the quantum is reported, because the empty-valley assumption is thin there.
- **Content keys.** A content key is exact discrete data and never a rounded float: int8 allele rows (rows sorted
  lexicographically, so the reference allele 0 sorts first), index paths, integer counts or sample names.
- **Truncation.** When a ranked list is cut (beams, shortlists, screens), the class the cut falls in is kept whole, up
  to twice the nominal size.
- **Threshold bands.** A value within a band of a threshold, for example a probability of exactly 1/2 or a
  contribution of exactly 0, is given the threshold's fixed side. That side is usually the conservative one: not a
  supporter, not released.
- **Abstention.** Where a stage can leave a result unresolved, a tie abstains instead of choosing. This covers
  pedigree winners, completion links, neighbour ties whose released alleles differ, unidentifiable fills, multimodal
  sites and export calls at equal totals.
- **Sample order.** The run's sample axis is ordered by sample name at the evidence stage, and reports and the export
  restore the input order. "Lowest sample index" in the rules below therefore means first by name. Outputs do not
  depend on the input order, but they can change if samples are renamed.
- **CPU independence.** Discovery's HDBSCAN orders links of equal weight by a fixed pseudo-random key of the
  canonical record indices (splitmix64 of the record pair), so the result does not depend on the CPU's sort kernels. This is a
  content-key convention under review: which equal-distance merge comes first changes the clusters, and so the
  candidate rows offered to the search. Logarithms and exponentials in tie-relevant kernels use the scalar C library (no SIMD variants, and
  `NUMBA_DISABLE_INTEL_SVML` is set at import). No tie-relevant decision uses BLAS.

## Conventions by stage

The tables list the decisions whose ties reach released outputs. Each table gives the alternatives that tie and the
convention that decides. *Alternatives*: **relabel** = pure relabellings (same result); **equal** = different
hypotheses with exactly equal support; **near** = within one quantum; **preference** = a parsimony rule applied
before content.

### Evidence ([evidence.md](evidence.md))

| decision | alternatives | convention |
| --- | --- | --- |
| the run's sample axis | permutations (relabel) | sample-name order; the export restores the input order |
| read-model fit and calibration | equal optima | first start in a fixed order; boundary dispersion 0; the simplest model in candidate order |
| histogram modes, minor allele, QC thresholds | equal | first (lowest) bin; ALT; fixed inclusive or strict sides |

### Discovery ([discovery.md](discovery.md))

| decision | alternatives | convention |
| --- | --- | --- |
| a sample's diploid state in a fit (A step) | rows identical over the sample's observed sites (equal); near | first state in a fixed order: row pairs row-major, then (row, unknown), then (unknown, unknown) |
| a founder allele no assigned sample observes (H step) | both alleles cost 0 (equal) | keep the current allele, which is the seed's convention: REF for subtraction seeds, first argmax, population ALT for dosage seeds. Calls withhold these alleles |
| the representative panel of each K, archive order and width cut | panels differing only at unobserved alleles (equal); near | NLL tie class, then panel bytes; the width cut falls inside a class by content |
| the released panel across K | cavity scores within a quantum (near; rare) | score class, then **smaller K** (preference), then panel bytes |
| which misfit samples seed K + 1 children | equal excess (equal); near | excess class, then more observed sites, then sample index; tie groups are kept whole up to twice the budget |
| K = 1 and dosage seed rows | consensus exactly 1/2; equal-size clusters | population ALT fraction (REF if also in the band); first argmax (REF); HDBSCAN label order |
| soft-residual births | probability exactly 1/2; exact distance ties | ALT at exactly 1/2; lowest row; more supporting samples, then row bytes |
| held-out mean-field start | mirror modes with equal ELBO | first converged start of H, 1 - H, 1/2 within a quantum |
| a carrier counted as a supporter | contribution exactly 0 | not a supporter (conservative) |

Gauge moves, replacements, deaths and "improved" tests keep the incumbent unless the challenger is better by more
than a quantum. Candidate lists are ordered by NLL class, then K, then panel bytes.

### Local selection ([local_selection.md](local_selection.md))

| decision | alternatives | convention |
| --- | --- | --- |
| the bank panel of each round (BIC) | BIC within a quantum (near); equal NLLs | BIC class, then NLL class, then **fewer founders** (preference), then canonical key (also inside wide classes; never admission or hash order) |
| the path-search start | refits of one source within a quantum | score class, then NLL class, then fewer founders, then key; the last non-empty source's winner starts the search |
| proposals refitted per kind (top 8) | screened scores within the tolerance | the eighth's tie class joins the refits, up to 12; a cut still inside the class goes by panel bytes, then generation order |
| the round winner | different panels within the tolerance | class, then **fewer founders**, then panel bytes, then trial order (drop, merge, add) |
| stopping, keeping the previous round's outcome | gains within the tolerance | keep the incumbent unless better by more than the tolerance |
| the context code (founder claims) | equal-cost matchings | the cheapest maximum-cardinality matching of founder claims (order-free in count and code) |
| supporters and carriers | exact symmetric cells | not favourable, not a carrier |
| a site's MAP number of unknown copies | count masses within 2^-20 of the largest | the smaller count |

### Assembly ([assembly.md](assembly.md))

| decision | alternatives | convention |
| --- | --- | --- |
| accept a join move or keep the incumbent | LL gains within about 16 quanta | keep the incumbent |
| the tier-1 or tier-2 join move | improving trials within a quantum; equal LL | LL class, then the trial's sorted path tuples or panel key |
| the death candidate tested at a count step | near-equal V(P - f); paths differing only where no sample is informative | pure-mosaic candidates first (preference), then V class, then path tuple; only the first member is count-tested |
| pool births and seeds | equal gains; integer coverage ties | gain class, then the new path tuple; the screen cut keeps its class |
| paths kept per beam endpoint | scores in one class; equal sums | class, then lexicographic path; the cut class is kept up to twice the quota |
| a completion link | runner-up margin below 1e-12 | abstain |
| an abundant route | sum of squares exactly at the threshold | not abundant (conservative) |
| the best atomic neighbour | equal qualifying counts and scores | most qualifying, then score class, then first; abstain if the tied neighbours release different alleles |

### Founder refinement and the count rule ([founder_refinement.md](founder_refinement.md))

| decision | alternatives | convention |
| --- | --- | --- |
| the refit decided at a count step (which founder is deleted, which seed's K + 1 refit is compared) | refits within the refinement quantum (near; duplicate founders) | F class, then the content of the deleted founder's (or copied seed's) row path, then index. Every other class member is shadow-scored and recorded |
| zero-gain cleanup | keep or delete a founder no sample uses | delete, in canonical order, while every per-sample loss is at most 2^-14 (declared model change of release 2.1) |
| accept a larger panel | margin exactly 0 | strict test, so an exact zero keeps the **smaller panel** (preference) |
| forward-first or reverse-first fit | F within a quantum (mostly identical panels) | F class, then panel key, then forward |
| the best-F panel per K; count cycles | fits within max(1e-4, quantum) | first seen; a cycle keeps the smaller K, then the incumbent |

### Painting and founder refit ([painting.md](painting.md), [founder_refit.md](founder_refit.md))

| decision | alternatives | convention |
| --- | --- | --- |
| where a painted switch sits | a switch anywhere in a stretch of identical emissions (equal) | within 2^-20: **fewest track changes** (preference), then smallest index. The switch lands on the leftmost tied edge |
| track order after an edge | pure homologue relabelling; equal-score crossings | fewer allele-visible changes, then the current order. This order becomes the released genotype order of roots and one-parent children |
| label-continuity relabelling | the same transposition at two links in a stretch of identical rows | gain class, then leftmost link, then pair order |
| founder refit: which row's allele flips at a site | two rows painted only together (equal) | per site, flips then swaps in rounds judged against the round-start state; interacting moves in tie-class order (2^-16), then content; thresholds 5 nats (swaps 10) |

### Pedigree ([pedigree.md](pedigree.md))

Abstention is the main rule. M0/M1/M2 states, configuration identities and informative contigs whose margins are
within max(512 ULP, 2^-15) stay unresolved. The fixed rules are listed below.

| decision | alternatives | convention |
| --- | --- | --- |
| replacement row for a child whose row closes a cycle | tied feasible rows | the unique winner of the highest feasible state; on a tie the next lower state (**fewer parents**); children whose weakest links tie are reported as conflicts |
| greedy insertion order | equal role and confidence | role, then confidence class, then child index (name) |
| screen parents per child (top 20) | utilities within 2^-32; identical candidates | class, then parent index (name); the cut class is kept up to 40 |
| Parent1 / Parent2 slot order | slot labels (relabel) | lower sample index (name) first. Family phase uses the slot as the child's homologue |
| the evidence scale | flat or multimodal objective | NLL class, then the scale nearest 1 |

### Family phase ([family_phase.md](family_phase.md))

The family MAP solver and the genotype-preserving polish share these rules.

| decision | alternatives | convention |
| --- | --- | --- |
| accept a block, move or interval change | score differences within the gain quantum (2^-17 for the MAP solver; the polish's minimum gain) | keep the current configuration |
| selector and frame Viterbi paths | a switch anywhere in an uninformative stretch (equal) | back-pointers within 2^-29 stay (fewest switches); the final state is the current one when within 2^-29 of the best, else the lowest |
| a child's genotype at a site (MAP solver) | within 1e-12 | keep the current genotype |
| the starting configuration | every exact tie (the current state wins ties) | frames 0 (the scaffold orientation); children in generation order, each at its exact block optimum given its parents' summed genotypes; the polish starts from frame 0 with selectors in state 0 |
| update order | blocks of one colour share no variable | conflict-graph colours with samples in canonical (name) order, then relabelling scans, compound moves and site moves in canonical order; the polish colours by degree, then sample index. The result does not depend on the thread count |
| frames the data leave free | relabellings | state 0, the painting's track order; samples with fewer than two observed parents keep frame 0 in the polish |

### Recombination and export ([recombination.md](recombination.md), [usage](../usage.md#export))

| decision | alternatives | convention |
| --- | --- | --- |
| which orientation tract is corrected | equal degree explaining the same artifact | higher degree, then sample index (name), then start; flip only if better by more than 2 nats |
| crossover calls | uninformative positions | intervals between supported markers (interval-censored, no point choice) |
| genotype allele order, phase sets, founder columns | relabellings | displayed track 0 first; phase set = component start; component then row order; '.' at equal totals |

## What a run records

Every stage that decides with tie classes stores `tie_events` in its records. This is a `maths.ties.TieCounter` record
of the following counts for each decision kind (named `<stage>.<decision>`):

- `decisions`: decisions taken;
- `inside_tie`: decisions taken by the convention because the best class, or a truncation cut, held several
  alternatives. For a kind ending in `band` it counts values inside the threshold band instead;
- `wide`: tie classes wider than four quanta;
- `boundary`: gaps near a quantum.

Further fields record ambiguity:

- discovery block summaries: `tie_quantum` (the block's NLL quantum). `ambiguous` is set when the selection was taken
  inside a tie: another K's representative scores within the selected score's class, or another panel lies in the
  first NLL class of the selected K. `tied_alternatives` is the number of such alternatives, and `score_margin` is
  the selected score minus the best other K's score;
- local-selection block summaries: `incumbent_margin`, the new search's objective minus the previous round's;
- refinement records:
  - each count decision records its `tied_candidates`, their `tie_decisions` and `tie_decisions_agree` (whether every
    tied refit gives the same accept-or-reject decision);
  - each component records `ambiguous`, `count_cycle` and `cycle_undecided`;
- count calls of the assembled components: `ambiguous` (some decision's bootstrap band contains zero, the step cycled,
  or the restored panel failed its own deletion test) and `count_cycle`.

These fields are reported only; no stage reads them. When the run completes, `ties/decisions.csv` sums the
`tie_events` per contig, record group and kind, and `ties/summary.json` holds the per-stage totals (tie-class
decisions separately from band comparisons) and the ambiguity counts above, including how many tied refits disagree.
The run log ends with a one-line digest. The [usage guide](../usage.md#outputs) lists the files.
`HAPLOTYPES_TIE_CENSUS` (a directory) switches on the diagnostic valley census that placed the quanta
([configuration.md](../configuration.md)).

## Where the choice is known to matter

Measured on the TroMau benchmark run and three simulated datasets during the release 2.1 validation:

- **Discovery.** Reversing every content convention at once changes 10.4% of TroMau blocks. About half of these end
  at an equal score, and the rest are split between better and worse fits. On TroMau, 196 of 9,979 released block
  panels have a tied alternative.
- **Refinement.** In 20 of 3,753 tied count decisions over four datasets, the tied refits disagree on K. The content
  rule decides these.
- **Pedigree.** No winner was decided inside a tie: ties abstain.
- **Painting relabelling.** 4 of 21 rankings were decided inside a tie.

Elsewhere the counts show how often a convention decides, not whether the alternative would differ.

The plan for release 2.2 is to carry every tie resolution forward as a candidate and let the block score decide,
instead of choosing by convention at the tie. The review starts from these conventions:

- the discovery representative and its search conventions;
- the join content choices, where the tied members of a death candidate are not shadow-tested;
- the refit choice at count steps;
- the Viterbi switch positions of painting and family phase;
- dependence on sample names: Parent1 is the lower name, and that fixes which child homologue is credited to which
  parent.
