# Founder refit

Assembly never re-calls an allele: a chromosome founder row copies the alleles of the block-local rows on its
path. Where block-level model selection merged two founders that differ at a few sites, the row of the founder
without its own local row carries the other founder's alleles there, for all of its carriers. The founder refit
(stage `founder_refit`, entry point `founder_refit.run`) performs one conditional M-step of the founder alleles
given the fixed [painting](painting.md): each called allele of each row is re-called from the reads of the samples
painted as carrying that row. It runs genome-wide after pedigree inference, which it does not read, one chromosome
per worker. Its released alleles are the founder panels of evaluation and export, and [family phase](family_phase.md)
decodes its scaffold from its inference alleles. Row paths, founder counts and the painting are unchanged, and
painting and pedigree keep reading the frozen assembly snapshot, so nothing feeds back into carrier-state evidence.

## Inputs and outputs

| | record | content |
| --- | --- | --- |
| reads | `painting` | the `ChromosomePainting` with its painted panels (assembly components after label continuity): released labels and the panels' released and inference calls |
| reads | `evidence` | normalised genotype likelihoods and the positive-depth mask through the evidence's founder view: samples the sample-quality QC keeps out of the contig's founder construction are uniform and unobserved there |
| writes | `identity/founder_refit` | `FounderRefitConfig`, the identities of the `painting` and `evidence` records, code identity (founder refit and painting packages) |
| writes | `founder_refit` | per component the refit released alleles and the changed cells (row, site, accepted gain) |
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

for dosage g in {0, 1, 2}, with ε = the painting's `PaintingConfig.robustness_epsilon = 0.01` (the refit maximises
the painting's own emission given the painted rows; P(reads | g) is the evidence stage's normalised genotype
likelihood). The objective of row k is then T[k, k, 2 x_k] plus T[min(k, p), max(k, p), x_k + x_p] over its
partners p with a called allele; partners with an unknown allele contribute nothing.

Coordinate ascent updates each site independently with `FounderRefitConfig.coordinate_sweeps = 3` sweeps. Each
sweep first tries every single-row flip (rows in order), then every swap of two rows (pairs in order), and applies a
move immediately when its gain is positive and reaches the prior of every cell it changes:
`FounderRefitConfig.minimum_gain_nats = 5.0` for a flip (one cell) and 2 x 5 = 10 nats for a swap (two cells). The
5 nats are a fixed prior against change per allele (the assembled allele keeps prior odds of about 150 to 1), so a
swap is accepted exactly when it is the more probable of the two configurations under independent per-cell priors.
A swap exchanges the different alleles (x_a != x_b) of two rows painted together at the site (some scoring sample
is painted with the pair {a, b} there). It corrects an allele carried by the wrong one of the two rows: it keeps the
genotype of every sample painted with that pair, so it costs no likelihood there but produces phase errors, and a
single flip of either row loses likelihood on those samples; only the swap undoes it. Its gain is the sum of each
row's change on its other carriers. Two rows with equal alleles are never flipped together, and rows that are not
painted together are only ever changed one at a time, each on its own 5-nat evidence.

Unknown alleles are never filled: only called cells can change. The refit runs on the released calls, so
completion fills are refit under the same rule as assembly calls. Sites outside the span of the component's
painting bins have no painted carriers and keep their alleles.

## Product and outputs

`FounderRefit(contig, sample_ids, components, excluded_samples)` (`founder_refit/refit.py`) holds one
`ComponentRefit(panel, changed_rows, changed_sites, gain_nats)` per painted component and the samples whose reads
the sample-quality QC keeps out of this contig's founder evidence. Each component keeps two matrices:

- `panel.released`: the refit alleles. They are founder stage `founder_refit` of evaluation and the founder tracks
  of export.
- `panel.inference`: the refit alleles restricted to the frozen snapshot's called cells (−1 where the snapshot is
  unknown). Family phase builds its scaffold from them.

Every other panel field is the painted panel's. The changed cells list each released cell whose final allele
differs from its painted value, with the gain of its last accepted move. The record stores only the refit released
alleles and the changed cells; the panels are rebuilt on load from the painted panels.

`OUTPUT/founder_refit/summary.csv` has one row per contig: `components`, `founder_rows`, `sites`,
`called_alleles`, `changed_alleles`, `changed_inference_alleles` (changed cells that are called in the snapshot),
`changed_sites`, `median_changed_gain_nats` and `excluded_samples`. The log line
`[founder refit <contig>] m of n called alleles changed at s sites` also names the QC-excluded samples.

## Implementation and cost

- **Structure.** One chromosome per worker of a chromosome executor; the number of concurrent chromosomes is
  limited by a per-chromosome byte estimate (the decoded evidence plus twice the decoded painting and 2 GiB of
  working space). Inside a chromosome, sites are processed in parallel tiles of 1024 sites, each tile owning one
  K × K × 3 table.
- **Cost.** Per site: O(N) to fill the table, O(K^2) per sweep for the single flips (each row's objective sums over
  its K partners) and for checking the K (K − 1) / 2 pairs, plus O(K) for each swap candidate (a pair painted
  together at the site, at most min(N, K (K − 1) / 2) of them); O(L (N + K^2 + P K)) per component with three
  sweeps, for P co-painted pairs per site. Memory: the K × L refit and gain arrays plus one K × K × 3 table and one
  K × K co-painting mask per tile.
- **Resume.** `identity/founder_refit` is bound before a contig's record is tested, as in the other stages: a
  finished contig is reused only under the same refit settings, upstream records and code, and anything else is
  refused (use a new output root). `complete/founder_refit` is marked once every contig is recorded.
- **Determinism.** Sites are independent and each site's sweeps run serially in a fixed order, so results do not
  depend on the thread count or the tiling.

## Interpretation and limitations

- The refit is conditional on the painting: carriers are those the painting names, and the painting is not
  recomputed with the refit alleles (no repaint or second refit).
- A row receives evidence only where its carriers are painted on it with both tracks named. Rows of locally rare
  founders whose carriers are painted on another row, or on pooled tracks, receive no carrier evidence and keep the
  alleles they borrowed in assembly.
- On real data the refit follows the reads, including at artefactual sites (paralogues, excess heterozygosity); no
  site-quality mask is applied here.
- Samples excluded by the sample-quality QC on a contig contribute nothing there, as in founder construction; they
  are still painted and phased.
- Pedigree results do not depend on the refit (pedigree reads the painting and the frozen snapshot).
