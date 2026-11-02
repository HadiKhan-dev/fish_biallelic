# Numerical ties and conventions

A decision names its compared quantity, units, direction, scale, quantum and exact content key. A tie class groups neighbouring values; it does not replace a model's incumbent preference, threshold band or ambiguity rule. The [generated conventions reference](../reference/conventions.md) contains the shared quanta, named policy call sites and random stream registry. Stage methods explain each decision's scientific preferences.

## Classes, bands and strict improvements

`core.ties.tie_classes` sorts exact values, then starts a new class when the gap from the preceding value exceeds the quantum. This is consecutive-gap single linkage: membership is not distance from the first element or the best score. NaN values are refused; equal infinities can remain together. Within a class the order is secondary scientific key followed by exact content. If its span exceeds `WIDE_CLASS_QUANTA` times the quantum, value order takes precedence within that wide class. A class can therefore span more than one quantum without making all members exchangeable.

The shared kinds are `nll`, `h_step`, `improvement`, `sample`, `cell` and `probability`. Their quantum is a power of two rounded upward from the larger of the kind's floor and relative factor times the absolute decision scale. Floors derive from `NOISE_MOTION`, `NOISE_MARGIN` and `GL_NOISE`. Assembly kinds use `ASSEMBLY_QUANTA` and their recorded motions. A fixed policy uses its supplied positive finite width unchanged. Values and source expressions belong in the generated reference, not copied defaults in methods prose.

`improves` accepts a gain only beyond the quantum; equality is not improvement. `threshold_band` distinguishes values below, within and above the inclusive uncertainty band. These are scalar rules and do not introduce a content convention. Boundary census records gaps near the quantum using `BOUNDARY_GUARD`; width and boundary counts diagnose sensitivity, not grounds to silently enlarge a tolerance.

## Content order and reversal

Content keys are exact discrete data: integer/boolean arrays, bytes, strings or tuples of such values. Floating values are not content keys. Scientific secondary preferences, such as founder count or minimum switches, remain separate and never reverse. Equal keys require an argument that the represented alternatives are exchangeable. Duplicate exchangeable proposals keep the smallest enumerated representative in either convention; changing which duplicate first supplied a description must not change the released claim.

`TieSettings.reverse` reverses arbitrary content conventions. Its `keep` tuple holds named discovery conventions at the default, with `all` holding all of them. Both affect the derivation; the optional census sink does not. The setting is supplied explicitly through typed stage boundaries. Science reads no environment switches for tie reversal or census. Canonical input sample order uses sample names; output adapters restore the required input order.

A reversal test runs the same boundary fixture with default and mirrored conventions, compares the scientifically invariant output, and records changes in arbitrary representatives and decision operands separately. It must exercise a real tie; identical untied runs are not evidence that reversal works. Fixtures near each cutoff include both sides and the boundary itself when representable. The [testing guide](testing.md) describes comparison and register receipts.

## Stage-specific limitations

Family phase keeps the current state on tied local moves, preserves gauge continuity at interior bins, and uses the terminal gauge convention at the final bin. Release visits samples in topological depth and ascending sample-index order in both convention solves; storage casts follow float64 chain tests. Those rules are not replaced by a generic sorted class.

Painting snaps the relevant emission representation to its named grid before required algebraic operations. Pedigree's screen softmax has no clipping cutoff; a tolerance used after a bridge solve is a post-check, not the root solver's stopping rule. Relatedness updates its member sum before adding parent terms. These forms and the other declared numerical choices are documented where they act in the methods; a shared helper does not authorise reassociation or a broader identity exception.

## Numerical ledger

The ledger lists the declared numerical normalisations of this release: places
where one principled arithmetic form replaces several forms, or a declared order
replaces an order chosen by workers, memory, object identity or the environment.
A ledger entry never changes a model, objective, threshold, candidate set, search
order or budget. A decision that moves only because of an entry is a reported
boundary flip with its operands and downstream effect; it is never hidden by a
tolerance. The owner approved entries L1–L15 and their scoped bounds. Measured flip
counts and margins are reported with the release in the [changelog](../../CHANGELOG.md).

| Entry | Category | Form | Where it acts |
|---|---|---|---|
| L1 | Formula unification | One Haldane recombination fraction, computed with `expm1`, for every map distance | [genetic map](../methods/genetic_map.md#ec27), [phase edge rates](../methods/phase.md#fp02), [pedigree selector switches](../methods/pedigree.md#ps04) |
| L2 | Summation order | One declared order (sample, natural or C order) for sums whose order otherwise followed worker counts, memory-sized batches or array memory order; the painting estimator's parameters are compared on their 2^-6 log grid | [painting estimator](../methods/painting.md#pr08), [phase release](../methods/phase.md#fp15), [phase storage routes](../methods/phase.md#fp22), [pedigree inputs](../methods/pedigree.md#pd02), [genome likelihood](../methods/pedigree.md#pd04), [junction schedules](../methods/pedigree.md#ps19) |
| L3 | Tie order | A declared, stable content order decides exchanges between exact duplicates | [assembly ties](../methods/assembly.md#u14), [refinement ties](../methods/refinement.md#rs09) |
| L4 | Dtype at a test | Family phase's chain test uses float64; float32 only for storage | [phase posterior and release](../methods/phase.md#fp15) |
| L5 | Deterministic replacement of a memory-dependent choice | The shadow-domain proof is attempted by a dimension rule, never by available memory | [shadow-envelope proof](../methods/assembly.md#aj35), [refinement count](../methods/refinement.md#rc12) |
| L6 | Label numbering | Family numbers are canonical dense ranks ordered by each family's smallest member | [relatedness release](../methods/relatedness.md#rl19) |
| L7 | Deterministic replacement of an object-identity choice | A stall's second count step runs only when the extension pool's content differs; `source_decision` names follow the decision sequence | [join records](../methods/assembly.md#aj13), [count steps](../methods/assembly.md#aj26) |
| L8 | Environment flags | Tie reversal and the census are explicit settings passed through typed boundaries | [tie settings](../methods/core.md#ec33), [pedigree ties](../methods/pedigree.md#pd11), [refinement](../methods/refinement.md#rs13) |
| L9 | Deterministic replacement of a scheduling-dependent choice | The window-work witness is the first exhausting template in the count pass's declared order | [refinement work account](../methods/refinement.md#rs07) |
| L10 | Formula unification | Founder refit qualifies and re-checks two-row moves with one direct two-row gain | [refit ascent](../methods/refit.md#fr02) |
| L11 | Formula unification | The selection context refit's inter-bin distance is computed through the map's interval distance | [context refit](../methods/selection.md#ls26) |
| L12 | Formula unification | Pedigree emissions use `pow(x / max, η)`, skipping the power at η = 1, on every route | [child emissions](../methods/pedigree.md#ps10) |
| L13 | Formula unification | Relatedness predicts and rescores a family's log-likelihood with one function of the member sum | [relatedness family terms](../methods/relatedness.md#rl06) |
| L14 | Formula unification | The template linker is compiled without fast-math reassociation; it acts only when templates are rebuilt | [template linker](../methods/simulation.md#ou21) |
| L15 | Output representation | Unphased sample genotypes are written in ascending allele order, missing first, instead of 3.5's internal order | [sample phase VCF](../methods/export.md#ou05) |

The reversal test and the comparison of a ledger entry are different tools. The
reversal test asks whether an arbitrary convention moves a scientific claim; the
ledger names an approved change of arithmetic and requires its receipts.

## Random streams

Use a named stream with an explicit realised seed or the declared content-derived bootstrap key. Preserve the generator type, draw shape, draw order, contig order and retry behaviour. A fixed seed alone does not establish identity after batching changes. Randomness is part of the generating model or resampling procedure; execution scheduling must not select a different stream. The stream catalogue is generated beside the tie constants.
