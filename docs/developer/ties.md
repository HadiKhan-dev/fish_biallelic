# Numerical ties and conventions

A decision names its compared quantity, units, direction, scale, quantum and exact content key. A tie class groups neighbouring values; it does not replace a model's incumbent preference, threshold band or ambiguity rule. The [generated conventions reference](../reference/conventions.md) contains the shared quanta, named policy call sites and random stream registry. Stage methods explain each decision's scientific preferences.

## Classes, bands and strict improvements

`core.ties.tie_classes` sorts exact values, then starts a new class when the gap from the preceding value exceeds the quantum. This is consecutive-gap single linkage: membership is not distance from the first element or the best score. NaN values are refused; equal infinities can remain together. Within a class the order is secondary scientific key followed by exact content. If its span exceeds `WIDE_CLASS_QUANTA` times the quantum, value order takes precedence within that wide class. A class can therefore span more than one quantum without making all members exchangeable.

The shared kinds are `nll`, `h_step`, `improvement`, `sample`, `cell` and `probability`. Their quantum is a power of two rounded upward from the larger of the kind's floor and relative factor times the absolute decision scale. Floors derive from `NOISE_MOTION`, `NOISE_MARGIN` and `GL_NOISE`. Assembly kinds use `ASSEMBLY_QUANTA` and their recorded motions. A fixed policy uses its supplied positive finite width unchanged. Values and source expressions belong in the generated reference, not copied defaults in methods prose.

`improves` accepts a gain only beyond the quantum; equality is not improvement. `threshold_band` distinguishes values below, within and above the inclusive uncertainty band. These are scalar rules and do not introduce a content convention. Boundary census records gaps near the quantum using `BOUNDARY_GUARD`; width and boundary counts diagnose sensitivity, not grounds to silently enlarge a tolerance.

## Content order and reversal

Content keys are exact discrete data: integer/boolean arrays, bytes, strings or tuples of such values. Floating values are not content keys. Scientific secondary preferences, such as founder count or minimum switches, remain separate and never reverse. Equal keys require an argument that the represented alternatives are exchangeable. Duplicate exchangeable proposals keep the smallest enumerated representative in either convention; changing which duplicate first supplied a description must not change the released claim.

`TieSettings.reverse` reverses arbitrary content conventions. Its `keep` tuple holds named discovery conventions at the default, with `all` holding all of them. Both affect the derivation; the optional census sink does not. The setting is supplied explicitly through typed stage boundaries. Science reads no environment switches for tie reversal or census. Canonical input sample order uses sample names; output adapters restore the required input order.


## Stage-specific limitations

Family phase keeps the current state on tied local moves, preserves gauge continuity at interior bins, and uses the terminal gauge convention at the final bin. Release visits samples in topological depth and ascending sample-index order in both convention solves; storage casts follow float64 chain tests. Those rules are not replaced by a generic sorted class.

Painting snaps the relevant emission representation to its named grid before required algebraic operations. Pedigree's screen softmax has no clipping cutoff; a tolerance used after a bridge solve is a post-check, not the root solver's stopping rule. Relatedness updates its member sum before adding parent terms. These forms and the other declared numerical choices are documented where they act in the methods; an equivalent reassociation needs the numerical review described in the [style guide](style.md). It does not authorise a new threshold or scientific rule.

## Random streams

Use a named stream with an explicit realised seed or the declared content-derived bootstrap key. Preserve the generator type, draw shape, draw order, contig order and retry behaviour. A fixed seed alone does not establish identity after batching changes. Randomness is part of the generating model or resampling procedure; execution scheduling must not select a different stream. The stream catalogue is generated beside the tie constants.
