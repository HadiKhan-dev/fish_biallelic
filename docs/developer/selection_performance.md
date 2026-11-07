# Selection performance at low depth

Selection has two distinct searches. The feedback-round candidate bank uses
discovery's fixed-K fits and cavity scores, then a BIC neighbourhood search.
The path search fits starts and drop, merge and add proposals under the copying
model. Each fit can perform GEM updates, frequency-only updates and context
snaps. Their definitions and retained stopping rules are in
[selection](../methods/selection.md); complexity exemptions are in
[costs](costs.md).

## Evidence and scope

A matched census covers all 1,647 chromosome-1 blocks in each of the initial,
L1 and L2 rounds at depths 5, 1, 0.5 and 0.1, with 320 simulated samples.
It reads committed block summaries and traces, including completed blocks in
an unfinished low-depth run. These are chromosome-specific measurements, not
whole-genome estimates. Mean values are:

| Depth | Initial selected K | L1 bank fitted starts | L2 bank fitted starts | L2 path rounds |
|---|---:|---:|---:|---:|
| 5 | 5.57 | 6.96 | 6.95 | 1.00 |
| 1 | 5.07 | 23.89 | 23.93 | 1.01 |
| 0.5 | 4.68 | 37.55 | 39.07 | 1.02 |
| 0.1 | 2.87 | 69.14 | 101.07 | 1.31 |

At 0.1 the L2 bank fits up to 1,452 starts in an exceptional block;
selected K alone misses this temporary work. The census has no selected-fit
update backstops, path-round backstops or proposal-cut overflows. Bank budgets
bind in five L1 blocks and eleven L2 blocks at 0.1, and none at the other three
depths. These endpoint receipts do not describe the termination of every
rejected fit.

Instrumented saved-input replays separate iteration work from bank size.
For matched block 200, the initial round performs 75, 151, 215 and 776 GEM
updates at depths 5, 1, 0.5 and 0.1. Its discovery starts have 6, 6, 6 and 2
rows. At 0.5 the L1 replay spends about 6% of its time constructing the bank;
copying-model fits dominate. At 0.1 this block also has a bank tail: L1 fits
735 bank starts. Its L1 copying fits perform 864 GEM updates, 77 trajectory
frequency updates and 1,295 context-snap frequency updates. The bank and
copying fits both contribute materially. Context description/matching takes
less than 1% in this example. Thus neither larger selected K nor the context
matching exemption alone explains the increase.

Finer counters on initial blocks 600, 1000 and 1400 show 11 exact single-bit
escapes in total at depth 5, versus 1,704 at 0.1. Frequency-only continuations
over the same blocks rise from 178 to 260. Much of the extra ordinary-GEM work
is therefore allele escape steps, rather than frequency-only convergence.
Context snaps add their own one-bit changes and frequency refits.

The replay restores each round's saved incumbent, earlier calls, context
proposal and exact initial/L1 memo. Missing likelihoods are neutralised through
the same logical evidence projection as production. An uninstrumented call on
the same inputs is the reference for diagnostic instrumentation. Replacing the
saved memo with a cold fit would test a different execution contract
([LS13](../methods/selection.md#ls13)).

## Measuring the driver

Retain both feedback-bank and path-search receipts:

- Discovery latent K, alternative panels, completed proposal rows, bank rows,
  number and sizes of starts, and distinct searched starts.
- Per-search round K, proposal counts by kind, whole-class refit admissions,
  overflows and actual refit count. A retained incumbent can carry an older
  trace; its trace is not a count of work attempted in the current round.
- Updates and termination of every executed fixed-K fit, including rejected
  trials; frequency-only continuations, trajectory frequency updates, context
  snaps and their frequency updates.
- Actual E-step passes and cache hits by statistics level. Calls to the state
  accessor include hits and must not all be counted as new passes.
- Separate inclusive timers for candidate-bank construction, bank fixed-K fits,
  cavity scoring, path fitting, context description/matching and release.
  Nested inclusive timers must not be summed as disjoint work.

Wall clocks and detailed diagnostic lists belong in the lab replay, outside
scientific decisions and checkpoint identities. Existing bounded result
summaries remain useful for broad censuses. Compare logical work before
attributing timings on different or shared hosts to algorithmic changes.

## Optimisation boundary

The measured work motivates testing joint exact allele escapes, reducing their
repeated frequency refits, and avoiding duplicate model searches in candidate-bank
construction.
Both can change the optimisation trajectory and selected panel, even if the
objective is retained; they require a separately validated scientific change.
Reducing budgets, relaxing tolerances or pruning by a proposal's starting
likelihood has no identical-output justification: refits can change alleles,
and current screening includes lower bounds and complete tie classes.

A memo of context descriptions is mathematically safe when keyed by the exact
panel, context rows, called masks and matching convention. Its possible gain is
limited by the measured matching fraction. It does not resolve the copying-fit
or candidate-bank costs. Preserve the incumbent/memo completion precedence,
whole-class admissions, conditional one-bit checks and context rescue witnesses.
