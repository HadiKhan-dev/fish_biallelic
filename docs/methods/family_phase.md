# Family phase mechanisms

The complete semantic-unit descriptions are in [phase](./phase.md).

## edge rates

See [FP02](./phase.md#fp02) for the model, decision order, parameters and limits.

## Resumable component work

See [FP23](./phase.md#fp23) for durable solve, canonical sample-batch release and component units, identity binding and stage-owned cleanup. No checkpoint is taken inside a solve.

## Read-class evidence

At the painting's confidence strength, a bin's emission for an ordered diplotype
is the tempered scalar emission times the painting's scaled read matrix. Reversing
the diplotype permutes both class axes. One-class runs and samples that bypass the
read state keep the scalar emission exactly.

For each sample with read sharing, family phase sums both orientations of every
flanking bin's diplotypes under independent uniform ordered-diplotype priors. The
resulting matrices carry a forward class vector from the first-site class law and
a backward vector from the end of the component. At a focal bin it contracts each
candidate's matrix between those vectors, averages the two homologue orders and
includes the matrix log scale. This gives single-bin likelihood ratios with the
same flanking evidence for all candidates and leaves the arbitrary homologue
frame symmetric. Using these single-bin terms in the family objective is a
composite likelihood. The exact sample-block solves optimise that objective,
which treats multiple candidate bins as separate terms. Both convention solves
and the release read the same projected emissions. See [FP01](./phase.md#fp01)
and the [painting read-state model](./painting.md#pr09).
