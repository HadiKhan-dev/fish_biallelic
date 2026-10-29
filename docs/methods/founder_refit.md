# Founder refit

Assembly never re-calls an allele: a chromosome founder row copies the alleles of the block-local rows on its
path. Where block-level model selection merged two founders that differ at a few sites, the row of the founder
without its own local row carries the other founder's alleles there, for all of its carriers. The founder refit
(stage `founder_refit`, entry point `founder_refit.run`) corrects the founder alleles conditionally on the fixed
[painting](painting.md): each called allele of each row is re-called from the reads of the samples painted as
carrying that row, and an allele the panel left unknown can be filled (release 2.18). It runs genome-wide after pedigree inference, which it does not read, one chromosome
per worker. Its released alleles are the founder panels of evaluation and export, and [family phase](family_phase.md)
decodes its scaffold from its inference alleles. Row paths, founder counts and the painting are unchanged, and
painting and pedigree keep reading the frozen assembly snapshot, so nothing feeds back into carrier-state evidence.

## Inputs and outputs

| | record | content |
| --- | --- | --- |
| reads | `painting` | the `ChromosomePainting` with its painted panels (assembly components after label continuity): released labels and the panels' released and inference calls |
| reads | `evidence` | normalised genotype likelihoods and the positive-depth mask through the evidence's founder view: samples the sample-quality QC keeps out of the contig's founder construction are masked as unobserved there (the likelihood arrays are shared) |
| writes | `identity/founder_refit` | `FounderRefitConfig`, the identities of the `painting` and `evidence` records, code identity (founder refit and painting packages) |
| writes | `founder_refit` | per component the refit released alleles and the changed cells (row, site, accepted gain, whether the cell was a fill); schema 2 (release 2.18; schema-1 records load with no fills) |
| writes | `complete/founder_refit` (run file) | genome-wide completion |
| publishes | `OUTPUT/founder_refit/summary.csv` | one row per contig ([outputs](#product-and-outputs)) |

Before refitting a contig the stage checks that its painting was computed from the contig's current evidence
record. `FounderRefitConfig` has no user-visible field ([configuration reference](../configuration.md)).

## Model

For every component, site j and founder row k, the refit chooses the allele x_k(j) in {0, 1} that maximises the
robust genotype log likelihood of the samples painted as carrying k at j, holding the painting and every other
row's allele fixed. The scoring set is the samples whose two released tracks are both named rows
(`SINGLETON_NAMED`) at the site and who have positive read depth there in the founder view. Pooled tracks
(`POOLED_EQUIVALENCE`, `LOCAL_ALLELE_EQUIVALENCE`), BACKGROUND, unanchored tracks and abstentions carry no
row-specific evidence; such a sample is skipped at the site together with its partner track.

Per site the evidence is collected once in a founder-pair table

    T[a, b, g] = Σ over scoring samples painted with the unordered pair {a, b} of log((1 − ε) P(reads | g) + ε/3),

for dosage g in {0, 1, 2}, with ε = the painting's `PaintingConfig.robustness_epsilon = 0.01` and P(reads | g) the
evidence stage's normalised genotype likelihood. This is a conditional correction of the alleles given the painting's
released labels, not the M-step of painting's own likelihood: since release 2.14 painting tempers its binned emissions
(an emission temperature and a design effect for correlated sites within a bin) and models panel-error runs, whereas
the refit scores one site at a time, where the carriers are distinct samples, with the untempered robust genotype
likelihood. The objective of row k is then T[k, k, 2 x_k] plus T[min(k, p), max(k, p), x_k + x_p] over its
partners p with a called allele; partners with an unknown allele contribute nothing.

Coordinate ascent updates each site independently with `FounderRefitConfig.coordinate_sweeps = 3` sweeps. A move
qualifies when its gain is positive and reaches the prior of every cell it changes:
`FounderRefitConfig.minimum_gain_nats = 5.0` for a flip (one cell) and 2 x 5 = 10 nats for a swap (two cells). The
minimum gain is charged on every accepted move against the cell's current allele: a first change of an assembled
allele needs 5 nats of evidence (the assembled allele keeps prior odds of about 150 to 1 against a single move, and a
swap is accepted when it beats the configuration it replaces under independent per-cell priors). A cell that has
changed can change again in a later round or sweep only on another 5 nats against its new allele, so the result is
a hysteresis rule, not exactly the maximum-a-posteriori configuration under fixed priors on the assembled alleles.

Each sweep has a flip phase and then a two-row phase (swaps and pair flips), and each phase runs in rounds (release
2.1). In a round every qualifying move of the phase is scored against the state at the start of the round and the
moves are taken in decision order: gain descending, gains within their tie quantum forming one tie class ordered by
content (the rows' canonical index order; [ties](ties.md)). Two gains summing n_i and n_j scoring carriers are tied
when they differ by at most max(2^-16, 2^-18 (n_i + n_j)) nats, and a tie class is a connected component of that
relation (release 2.18; computed in O(n log n) as the union of two interval relations, two sorts and union-find).
10^-6 relative noise in the genotype likelihoods, the 2.13 noise floor, moves one carrier's two log terms by at most
about 2·10^-6, so ties among moves with many carriers stay ties under that noise (until 2.17 the fixed 2^-16,
placed against 10^-9 noise; a 940-sample control changed a tie abstention into a call in 12 of 30 draws of 10^-6
noise, and in none now). The convention does not decide which moves are tied, and a move whose gain lies
between two tied gains cannot hide their tie (a component covers a contiguous range of gains; the convention only
orders the members of a class); a third move can still join two classes when it is tied with both. A component
wider than `WIDE_QUANTA` times max(2^-16, 2 x 2^-18 x its largest carrier count) is not a tie ([ties](ties.md)):
its members keep value order, each decided on its own. A move is
applied unless it interacts with a move already
applied in the round (it shares a row, or a row painted together with one of its rows at the site, whose score the
applied move changed); interacting moves are re-scored in the next round. A row changes at most once per phase and
sweep, and a sweep that changes nothing ends the site.

Two qualifying moves of one tie class that interact, where applying either one makes the other fail to qualify (one
direction suffices), are alternatives the data do not separate, typically two rows painted only together, either of
which could carry the allele their carriers show. Their rows' alleles at the site are released unknown (−1, `.` in
the exports) instead of being decided by row order, and the round is re-scored without them (release 2.2). These
cells are counted as `ambiguous_alleles`.
A swap exchanges the different alleles (x_a != x_b) of two rows painted together at the site (some scoring sample
is painted with the pair {a, b} there). It corrects an allele carried by the wrong one of the two rows: it keeps the
genotype of every sample painted with that pair, so it costs no likelihood there but produces phase errors, and a
single flip of either row loses likelihood on those samples; only the swap undoes it. Its gain is the sum of each
row's change on its other carriers. A pair flip (release 2.18) changes the equal alleles (x_a = x_b) of two rows
painted together: where both rows carry the wrong allele, their shared carriers' reads favour neither single flip,
because under the robust error floor a heterozygote fits a homozygote's reads barely better than the wrong
homozygote (with two shared carriers of 13 reads a single flip gains 0.08 nats, the pair flip 11.4), so only the
joint change reaches the genotype the reads show. Swaps and pair flips change two cells and need 10 nats. Rows that
are not painted together are only ever changed one at a time, each on its own 5-nat evidence.

The refit runs on the released calls, so completion fills are refit under the same rule as assembly calls. Sites
outside the span of the component's painting bins have no painted carriers and keep their alleles.

**Filled cells (release 2.18).** After the sweeps, an allele the painted panel left unknown is called from the same
evidence: over the row's scoring carriers whose partner allele is called (or who carry the row on both tracks), one
allele's objective must beat the other's by `minimum_gain_nats` (posterior odds of about 150 to 1 with even prior
odds, as there is no assembled allele to favour) and at least `FounderRefitConfig.fill_min_supporters = 2` of those
carriers must favour that allele on their own reads (their own log-likelihood difference exceeds the tie quantum).
This is the same minimum of two favourable samples that local selection and discovery require (`min_supporters`
= 2), not their whole release rule: one sample's reads, painted on one row, never release an allele the panel
withheld. All fills of a site are decided against the
post-sweep state and applied together, so they do not depend on each other or on row order; a carrier whose partner
is itself unknown gives no evidence, and cells the tie rule released unknown in this pass are not filled. Fills
change only the released alleles: the inference snapshot (family phase) keeps them unknown.

On the 2.17 validation inputs the fills reduced wrong-or-missing founder alleles on every simulated design (the
release's validation is in the CHANGELOG); a fill can still carry another founder's allele where the painting
names the wrong carriers, so fills are counted separately (`filled_alleles`).

## Product and outputs

`FounderRefit(contig, sample_ids, components, excluded_samples)` (`founder_refit/refit.py`) holds one
`ComponentRefit(panel, changed_rows, changed_sites, gain_nats, filled)` per painted component and the samples whose reads
the sample-quality QC keeps out of this contig's founder evidence. Each component keeps two matrices:

- `panel.released`: the refit alleles. They are founder stage `founder_refit` of evaluation and the founder tracks
  of export.
- `panel.inference`: the refit alleles restricted to the frozen snapshot's called cells (−1 where the snapshot is
  unknown). Family phase builds its scaffold from them.

Every other panel field is the painted panel's. The changed cells list each released cell whose final allele
differs from its painted value, with the gain of its last accepted move (for a fill, its margin) and whether it was a
fill. The record stores only the refit released
alleles and the changed cells; the panels are rebuilt on load from the painted panels.

`OUTPUT/founder_refit/summary.csv` has one row per contig: `components`, `founder_rows`, `sites`,
`called_alleles`, `changed_alleles`, `ambiguous_alleles` (called cells released unknown by the tie rule),
`filled_alleles` (unknown cells called by the fill rule),
`changed_inference_alleles` (changed cells still called in the refit's inference matrix: called in the snapshot and
not released unknown), `changed_sites`,
`median_changed_gain_nats` and `excluded_samples`. The log line
`[founder refit <contig>] m of n called alleles changed at s sites` also names the QC-excluded samples.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor; the number of concurrent chromosomes is
  limited by a per-chromosome byte estimate (twice the decoded evidence, as the founder view copies its
  likelihoods when sample QC excludes samples, twice the decoded painting and 2 GiB of working space). Inside a chromosome, sites are processed in parallel tiles of 1024 sites, each tile owning one
  K × K × 3 table.
- **Cost.** Per site: O(N) to fill the table. Per round: O(K^2) for the K single-flip gains (each row's objective
  sums over its K partners) and their carrier counts; O(1) per two-row candidate, a pair painted together at the
  site (at most P <= min(N, K (K − 1) / 2) of them), whose gain follows from the two rows' flip gains by replacing
  their shared samples' terms (swap(a, b) = flip(a) + flip(b) − T[a, b, 0] − T[a, b, 2] + 2 T[a, b, 1], as
  x_a + x_b = 1 for those samples; pair flip(a, b) = flip(a) + flip(b) + T[a, b, 0] + T[a, b, 2] − 2 T[a, b, 1]);
  O(n log n) to order the n qualifying moves. A phase has at most K rounds, so a site costs O(N + K^3 log K) per
  sweep; on every simulated and real case measured (K up to 32) a phase needed at most 3 rounds and at most 4 swap
  candidates qualified. The tie rule examines only pairs inside a tie class, at O(K) per tied interacting pair;
  only large tie classes (many moves whose gains lie within the carrier-scaled quantum, which at large N is much
  wider than 2^-16) can make those checks exceed the cubic bound, up to O(K^4) for flips and O(K^5) for swaps per
  phase (an accepted exception to the cubic rule). On every case measured a tie class had at most two members.
  Fills add O(K^2) per site (each unknown row's objective over its K partners).
  Memory: the painted labels expanded to the component's sites (int16, 4 N L bytes; about 1.3 GB at N = 1,000 on a
  330,000-site component), the K × L refit and gain arrays, and per tile one K × K × 3 table with K × K
  co-painting, carrier-count and support arrays.
- **Resume.** `identity/founder_refit` is bound before a contig's record is tested, as in the other stages: a
  finished contig is reused only under the same refit settings, upstream records and code, and anything else is
  refused (use a new output root). `complete/founder_refit` is marked once every contig is recorded.
- **Determinism.** Sites are independent and each site's rounds are decided by gain and content, not by the
  order rows are listed in, so results do not depend on the thread count or the tiling, and the convention
  reversal (`HAPLOTYPES_TIE_REVERSE=1`) only reverses the content order inside tie classes.

## Interpretation and limitations

- The refit is conditional on the painting: carriers are those the painting names, and the painting is not
  recomputed with the refit alleles (no repaint or second refit).
- A row receives evidence only where its carriers are painted on it with both tracks named. Rows of locally rare
  founders whose carriers are painted on another row, or on pooled tracks, receive no carrier evidence and keep the
  alleles they borrowed in assembly.
- On real data the refit follows the reads of the sites the evidence stage kept, including any artefactual sites
  the site filter did not remove (paralogues, excess heterozygosity); no additional site mask is applied here.
- Samples excluded by the sample-quality QC on a contig contribute nothing there, as in founder construction; they
  are still painted and phased.
- Pedigree results do not depend on the refit (pedigree reads the painting and the frozen snapshot).
