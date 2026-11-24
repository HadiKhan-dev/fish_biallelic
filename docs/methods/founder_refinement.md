# Founder refinement mechanisms

The complete semantic-unit descriptions are in [refinement](./refinement.md).

## acceptance

See [RS08](./refinement.md#rs08) for the model, decision order, parameters and limits.

## best of two

See [RS12](./refinement.md#rs12) for the model, decision order, parameters and limits.

## checkpoints

See [RC07](./refinement.md#rc07) for the model, decision order, parameters and limits.

## count continuation

See [RC17](./refinement.md#rc17) for the model, decision order, parameters and limits.

## count pass

See [RC09](./refinement.md#rc09) for the model, decision order, parameters and limits.

## diplotype exchange

A founder row is a path through the prepared rows of its blocks, so refinement can only choose among the splits of
a diplotype that some block already holds. When every carrier of a haplotype carries it with the same partner, a
block's genotype data fix the carriers' diplotype but not its split into two rows; the other split is then built
here, on each component of the final assembly after its last refinement fit, and accepted by the refinement's own
exact objective.

Decision order, per component with at least three founders:

1. Every sample's maximum path under the objective is painted over unordered founder pairs.
2. Row v is locked to u on a maximal run of sites where every sample whose path uses v is in the state {u, v},
   u != v; the samples using v anywhere in the run are its carriers.
3. An anchor is a founder w (not u, not v) that a sample using v at an end of the run pairs with v at the
   neighbouring site outside it. Every anchor gives one candidate.
4. The exchange row X, on the run's objective sites: X = v where u = v, X = 1 - w where u != v, uncalled where a
   needed call is missing. Where u != v the pair {w, X} has the dosage of {u, v}; where u = v != w it is one dose
   off and the objective charges it. No allele is estimated from evidence.
5. Copies are left out: a candidate whose X has no called disagreement with one of the K founder rows on the
   run's objective sites (the row v itself and rows other than the partner included; an uncalled allele agrees
   with anything) would put no new called allele content into the panel and is never scored.
6. Score every non-copy candidate on every sample. The whole-cohort gain must exceed the fixed-K acceptance
   quantum q = 2^-9 nats, both with all samples and after removing the most-gaining sample. The trial must also
   pass both carrier-support comparisons below. Commit the largest passing whole-cohort gain, breaking ties by
   v, start and w in ascending order. Repaint and derive the candidates again after every commitment.
7. After a pass with commitments the component gets the ordinary fixed-K fit of its level, with the polish when
   the run has one, on the panels that now hold the exchange rows; if the fit changes the panel the pass runs
   again. An equal-F family of the old panel is replaced by the family of that fit.

Let d_s be the change in sample s's maximum-path objective across the component, and C the carriers on the
current panel. Credit positive gains only from C and debit losses from every sample. The resulting score is
S = sum over C of d_s + sum outside C of min(0, d_s). An exchange requires both of the following.

- S must exceed log(2 (L−1) K (K−2)) + q, where L is the number of marker positions scanned and K is the number
  of founder rows. Given the current painting, an edit can be named by a boundary, its inside direction, row v
  and the rank of the anchor among founders other than v and its inside partner. The inside partner and the
  maximal locked run follow from the painting. The logarithm is the cost of choosing one such descriptor with
  uniform weights and equal starting weights for an edit and for keeping the panel.
- S − max(0, max over C of d_s) must exceed q. This allows any one carrier's entire improvement to reflect its
  private path. A genuine change seen in a single carrier can consequently be withheld.

The first comparison charges the search on the fitted painting. The second checks whether evidence remains in
other carriers. The objective uses clipped genotype likelihoods and maximum paths, so neither comparison provides
a calibrated probability of phase error. Accepted records include the carrier-backed gain, its deletion residual,
the descriptor cost and L.

A committed X is one appended prepared row in each block of its run: the source row on all four published arrays
outside the run's sites, X inside (probability the allele, 0.5 where uncalled, no supporter). Outside the run an
uncalled allele of the appended row is the same latent allele as the source row's, so the objective's
unknown-allele classes there are unchanged. The final product's record keeps the appended rows (panel, source row,
site range, alleles) in commitment order; the completion's own record keeps its panels.

Limits. The leave-one-out condition is a support safeguard, not an identifiability rule: the candidate and its
carriers are derived from all samples. The move cannot supply an allele that no row holds, reaches only runs with
a third row to anchor on, and leaves the part of an excursion outside the locked run. The objective can prefer a
split that is not the true one. Cost per round: one painting, O(S L K^2), and one exact score of the whole
component per candidate, with at most min(K - 2, 2S) anchors per run and at most K runs per site; rounds are one
per commitment plus one. A block holds at most 32,766 prepared rows (the int16 code of uncalled alleles): trial
panels are built in batches within it and a commitment beyond it is an error.

## emissions

See [RC02](./refinement.md#rc02) for the model, decision order, parameters and limits.

## fixed k search

See [RS11](./refinement.md#rs11) for the model, decision order, parameters and limits.

## gains

See [RS03](./refinement.md#rs03) for the model, decision order, parameters and limits.

## guard

See [RS10](./refinement.md#rs10) for the model, decision order, parameters and limits.

## memory

See [RC04](./refinement.md#rc04) for the model, decision order, parameters and limits.

## messages

See [RC05](./refinement.md#rc05) for the model, decision order, parameters and limits.

## objective

See [RS01](./refinement.md#rs01) for the model, decision order, parameters and limits.

## polish

See [RS14](./refinement.md#rs14) for the model, decision order, parameters and limits.
The polish orders candidate window paths by prepared-row bytes in the contig’s canonical allele coding. Row-specific unknown codes retain their identities.

The polish search retains last-problem beam answers and forward-oriented scalar gains within
one fixed-K fit; see [RS15](./refinement.md#rs15), [RS16](./refinement.md#rs16) and
[RS17](./refinement.md#rs17) for the keys and the label-dependent selections excluded from reuse.

## proposals

See [RS04](./refinement.md#rs04) for the model, decision order, parameters and limits.

## reuse inside a search

See [RS19](./refinement.md#rs19) for the restated window results after suffix exchanges, the relabelled message
updates, the shared sweep 0 of a fit's two searches and the reused standing-offer gains, with their exactness
conditions.

## shared switch guard

See [RC08](./refinement.md#rc08) for the model, decision order, parameters and limits.

## ties

See [RS09](./refinement.md#rs09) for the model, decision order, parameters and limits.

## window programme

See [RS05](./refinement.md#rs05) for the model, decision order, parameters and limits.

## window store

See [RC22](./refinement.md#rc22) for the component's window-result store, the polish answers its polish fits share,
concurrent count histories, the digest convention and the memory bounds.

## windows

See [RS04](./refinement.md#rs04) for the model, decision order, parameters and limits.

## work account

See [RS07](./refinement.md#rs07) for the model, decision order, parameters and limits.

## workspace

See [RC01](./refinement.md#rc01) for the model, decision order, parameters and limits.
