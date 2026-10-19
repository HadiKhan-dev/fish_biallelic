# Ties, released ambiguity and stated priors

A tie is a decision whose alternatives the data do not separate. Either they score exactly the same (for example two
panels that differ only at alleles no sample observes), or they differ by less than the rounding noise of the compared
sums. Release 2.2 handles ties by principle, not by convention:

- **No decision that selects between different biological answers depends on:**
  - array, row or sample order;
  - hash order;
  - the thread count or the CPU instruction set;
  - an arbitrary content-key order.

  Content keys remain only as canonical labels of the same answer, for example the row order of a panel.
- **Alternatives that more data can separate are carried forward** as a bounded, order-invariant set, and a later
  comparison decides between them:
  - discovery's tied clusterings and panels;
  - local selection's tied starts, bank members and refits;
  - assembly's tied joins and refits.

  A bound on such a set is itself order-invariant: a size cap on a canonical superset, with the overflow counted.
- **Ambiguity the data cannot resolve is released as such**, not replaced by a conventional point answer:
  - founder alleles that no carrier observes are written as `.`;
  - painted switches inside stretches where the founders are identical become intervals;
  - a phase set breaks where family evidence does not decide the orientation;
  - parent slots follow recorded sexes, and are unordered otherwise;
  - tied alternatives left at the end of a stage are flagged and released with the decision taken by a stated prior.
- **Stated priors decide the remaining exact ties.** They are model preferences with a biological or statistical
  reason, listed in [Stated priors](#stated-priors). They are never index or content order.

Two acceptance tests check this on every release candidate:

- **The noise loop** reruns the whole pipeline under input perturbations that should not matter:
  - 1e-12 and 1e-9 relative genotype-likelihood noise;
  - value-consistent noise;
  - permuted sample order, in memory and in the input file;
  - another thread count and another numerical environment.

  No biological outcome may change.
- **The convention reversal** reruns it with `HAPLOTYPES_TIE_REVERSE=1`. This reverses every remaining arbitrary
  convention: index order, content-key order, binary default states, sample-name order and keep-first. The stated
  priors and canonical labels are not reversed. Released outputs must equal the default run's, except inside released
  ambiguity. A residual difference marks a convention that still selects biology; the remaining ones are listed in
  [Remaining conventions](#remaining-conventions).

## Common machinery

`maths/ties.py` gives every stage one way to say "these values are tied".

- **Tie quanta.** Each kind of compared quantity has a power-of-two quantum. It sits far above the rounding of the
  compared sums and far below the first genuine differences, which a valley census of real and simulated runs
  measured. The kinds and their quanta are:
  - block NLLs and cavity scores: at least 2^-17, scaled with the block's null NLL;
  - H-step allele costs: 2^-18;
  - "improved" tests: 2^-14;
  - per-sample sums: at least 2^-23;
  - per-cell evidence: 2^-23;
  - probabilities: 2^-20.

  Stages with their own scale use the same construction, for example the refinement quantum
  max(2^-14, 2^-35 |F|) and the family-MAP gain quantum 2^-17.
- **Tie classes.** Values are sorted exactly, and consecutive values within a quantum chain into one class.
  - **Width guard.** A class wider than four quanta is reported and ordered by value.
  - **Boundary guard.** A gap near a quantum is reported.

  The members of a class are the tied set. They are carried forward, decided by a prior, or flagged. They are never
  ranked by content, except as a canonical label of an equal answer.
- **Threshold bands.** A value within a band of a threshold, for example a probability of exactly 1/2 or a
  contribution of exactly 0, is given the threshold's conservative side (a stated prior).
- **Reversal hook.** `maths.ties.reverse_conventions` reads `HAPLOTYPES_TIE_REVERSE`. The helpers `conv_key`,
  `conv_index`, `conv_order`, `conv_first`, `conv_argmin`, `conv_argmax`, `conv_state` and `conv_sorted` route every
  remaining convention through it. Without the variable they are the identity.
- **CPU independence.** Logarithms and exponentials in tie-relevant kernels use the scalar C library, and
  `NUMBA_DISABLE_INTEL_SVML` is set at import. No tie-relevant decision uses BLAS.

## Principled treatments by stage

| stage | 2.1 convention | 2.2 treatment |
| --- | --- | --- |
| discovery ([discovery.md](discovery.md)) | HDBSCAN merges of equal-weight links ordered by a content key; complete linkage on noise | every tie resolution of the residual clustering is carried as a candidate (level-wise condensed-tree groups and maximal-clique alternatives of the noise, with order-invariant bounds); the block score decides |
| discovery | NLL-tied panels decided by panel bytes | the tied panels are released with the block (`tied_panels`) and offered to local selection; archive beams keep whole NLL classes |
| discovery | founder alleles no carrier observes: the seed's allele | released unknown (`latent_unknown`) |
| discovery | tied A-step states of a sample: the first in row order | the meet of the tied states (a copy that differs between them goes to no founder) |
| local selection ([local_selection.md](local_selection.md)) | the path search starts from the last non-empty source's cavity-best panel | every source's cavity tie set is fitted and searched; outcomes go by objective classes, with fewer founders as the prior |
| local selection | bank, refit cut and residual ties by content and order | the BIC tie set is kept whole; whole tie classes are refitted (cap 16, overflow counted); residual ties give a consensus panel whose rows not shared by every alternative are withheld (`tied_alternatives`, `withheld_by_ties`) |
| assembly ([assembly.md](assembly.md)) | left-anchored batching | orientation-symmetric batching |
| assembly | join moves inside a log-likelihood tie class by path tuples or panel keys | the batch linker's path score decides before any label; painting-dependent proposals run under both traceback conventions and are merged |
| founder refinement ([founder_refinement.md](founder_refinement.md)) | tied refits decided by content | members that disagree on K take the smaller K (prior); otherwise the member whose margin most strongly supports the decision; remaining equal-F alternatives are flagged (`ambiguous_k`, `alternatives`) |
| founder refit ([founder_refit.md](founder_refit.md)) | the first acceptable move in row order | mutually exclusive tied moves release the affected cells unknown |
| painting ([painting.md](painting.md)) | a switch inside a stretch of identical founders placed at the leftmost edge, and the bins named by the prior | the switch is released as an interval (`SWITCH_INTERVAL`), and the other track is gated on the pair-consistent posterior; the per-sample segments are published |
| pedigree ([pedigree.md](pedigree.md)) | Parent1 is the lower sample name | parent slots follow recorded sexes (dam, sire); otherwise they are unordered (`ParentSlotBasis`) |
| family phase ([family_phase.md](family_phase.md)) | frames left free by the data took state 0 and the leftmost switch position | `frame_undetermined` marks frames another tied configuration sets differently, and the export breaks the phase set there |
| export ([usage](../usage.md#export)) | unobserved founder alleles written with the seed's value | written `.`, with counts in the manifest and the founder VCF header |

## Stated priors

These decide exact ties. They are not reversed by the reversal test.

| prior | where | reason |
| --- | --- | --- |
| fewer founders (smaller K) | discovery selection, the local-selection bank and round winner, founder refinement at an exact-zero count margin, count cycles | parsimony: at equal fit the data do not support the extra founder |
| fewer switches; fewer parents | the painting and family-phase Viterbi tie rule; the pedigree graph's replacement rows | at equal likelihood the recombination and transmission model already favours fewer events, because a switch probability is below 1/2 |
| keep the current state inside a tie class | search climbs, keep-incumbent tests, the family MAP's acceptance (a gain must exceed the quantum) | an ascent moves only on evidence, so the optimum is a fixed point and noise cannot move it |
| the conservative side of an exact threshold | release, supporter, carrier and unknown-count bands; the abundant-route test | at exactly the threshold, withhold rather than call |
| abstain | pedigree winners, completion links and neighbours whose released alleles differ, export `.` at equal totals | no released answer where the data give none |
| more observed sites and more supporting samples first | discovery's data-birth ranking and residual emission | the better-supported candidate is tried first; candidate sets are kept whole up to their caps |
| fewest unknown copies first; fewest flips first | discovery's A-step band; the local-selection site posterior | the explanation that needs fewer unknown copies or changes is preferred at equal cost |
| pure-mosaic deaths first; flips before swaps | assembly count steps; founder refit | simpler explanations first |

Canonical row orders, panel keys used as identities and the gauge of the two tracks are labels, not choices.

## Remaining conventions

On the release candidate the convention reversal still changes these released outputs:

| dataset | founder allele cells | genotypes | phase switches | meioses with other crossovers | Tier-B samples |
| --- | --- | --- | --- | --- | --- |
| TroMau, whole genome | 320,114 | 22,827 | 11,056 | 146 | 1 (status) |
| TroMau, three chromosomes | 3,225 | 1,361 | 1,713 | 0 | 2 (status) |
| std_n80 | 2 | 6 | 20 | 20 | 0 |
| stress_n80 | 0 | 0 | 13 | 26 | 0 |

They enter at the following decisions; everything downstream is inherited.

- **Discovery: content and sample order inside tie classes.** These are the central reversals of `maths.ties`.
  Through them the reversal changes 326 of 9,979 TroMau blocks and 26 of 1,028 on three chromosomes. The tie kinds
  inside those blocks are mostly the archive's NLL classes and the K = 1 seed bands, plus their overflow bounds.
  This is the largest residue, so the released founder panels of these blocks depend on a convention.
- **Local selection's bank, A-step band.** Discovery assigns tied A-step states their meet (the multiset
  intersection of the tied founder pairs). Local selection's bank keeps 2.1's rule, which takes the tied state in
  row order. A few entries remain: cavity-best and favourable-band ties, 6-8 blocks per round on TroMau. A
  fractional treatment is planned for 2.3.
- **Assembly.** A few joins (7 in 1,307 on the acceptance contigs) still follow the tier-1 proposal or evaluation
  path order and are not yet flagged. A few refinements reach different equal-F optima; these are flagged, and their
  alternatives are released.
- **Family phase.** The ascent under the canonical and mirrored schedules is not perfectly symmetric, so a few frames
  differ. On std_n80 that is 20 phase switches and 20 meioses with identical upstream.
- **Painting.** Track order is a label. The released painting (status, members, tracks unordered) is identical
  under reversal, switch intervals included.

## What a run records

Every stage that decides with tie classes stores `tie_events` in its records. For each decision kind (named
`<stage>.<decision>`) these count:

- `decisions`;
- `inside_tie` (the decision's class held several alternatives; for a kind ending in `band`, a value inside the band);
- `wide` and `boundary` (guard events).

The released ambiguity fields are:

- discovery: `ambiguous`, `tied_alternatives`, `score_margin`, `tied_panels`, `latent_unknown`;
- local selection: `tied_alternatives`, `withheld_by_ties`;
- refinement: `ambiguous_k`, `alternatives` and `prior` per count decision;
- founder refit: `ambiguous_alleles` (cells released unknown);
- painting: `SWITCH_INTERVAL` segments, and switch-interval counts in the painting summary;
- pedigree: `ParentSlotBasis`;
- family phase: `frame_undetermined` (bit 0 decoder, bit 1 displayed frame);
- export: `unobserved_founder_alleles`.

When the run completes:

- `ties/decisions.csv` and `ties/summary.json` sum the tie events per contig, stage and kind;
- `painting/<contig>/sample_segments.tsv` lists every sample's released painting, intervals included;
- the founder VCF and the export manifest count the alleles written `.`.

`HAPLOTYPES_TIE_CENSUS` (a directory) switches on the diagnostic valley census that placed the quanta, and
`HAPLOTYPES_TIE_REVERSE_KEEP` keeps named discovery conventions at their default under the reversal
([configuration.md](../configuration.md)).
