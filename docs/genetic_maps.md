# Genetic maps

Inference uses one genetic map for every stage: a fallback rate, `[genetic_map] rate_cm_per_mb` (5.0 cM/Mb, that
is 5e-8 Morgans per bp), and optionally a cumulative map, `[genetic_map] file`. A supplied map replaces the
fallback between its first and last knot on each chromosome it names; chromosomes it does not name, and the tails
beyond its knots, use the fallback rate, and the cumulative map stays continuous at both ends. Simulations configure
their generating map separately (`[simulation] generating_map`, `generating_rate_cm_per_mb`; see
[simulation.md](simulation.md)), so generating and inference maps never mix.

## Map files

The preferred format is a headerless, whitespace-separated PLINK/Beagle four-column `.map` file:

```text
chr1 marker1 0.0 1
chr1 marker2 5.0 1000001
chr1 marker3 15.0 2000001
chr2 marker1 0.0 1
chr2 marker2 2.5 1000001
```

The columns are `chromosome marker_id cumulative_cM position_bp`; marker IDs are ignored. The cumulative column is
in **centimorgans**: convert Morgans first. A headed three-column table is also accepted (header names are not
case-sensitive):

```text
chromosome position_bp map_cM
chr1 1 0.0
chr1 1000001 5.0
```

Chromosome names must match the variant file exactly (`chr1` and `1` differ), and positions must use the same
reference assembly and coordinates as the variant file; nothing is shifted. Within a chromosome, positions must be
strictly increasing (at least two knots) and cumulative cM finite and nondecreasing; zero may be a boundary knot.
Blank and `#` lines are ignored; reversed rows and duplicate positions are errors, never sorted or repaired. An
all-zero cumulative map means no recombination between its knots, not an absent map.

```toml
[genetic_map]
file = "data/cross.map"
rate_cm_per_mb = 5.0
```

From Python, `load_genetic_maps` also takes a mapping from exact chromosome name to a single-chromosome file,
which may be a SHAPEIT-style table with one of the headers `position COMBINED_rate(cM/Mb) Genetic_Map(cM)`,
`position rate(cM/Mb) map(cM)` or `physical_bp rate_cM/Mb cumulative_cM`; the cumulative column is authoritative
and the local rate is only checked:

```python
from haplotype_reconstruction.model.genetic_map import load_genetic_maps

maps = load_genetic_maps("cross.map", default_rate_cm_per_mb=5.0)
chromosome = maps.for_contig("chr1")
distance = chromosome.interval_morgans(500001, 1500001)   # 0.075 Morgans for the example above
```

## How the stages use the map

The map is piecewise linear in cumulative distance: a genetic distance is always G(right) - G(left), never one
endpoint's rate times the interval length. Equal knots define a cold interval, and sparse knots interpolate
linearly through their whole gap. Each model turns distances d (Morgans) into probabilities with one of two forms
(`model/genetic_map.py`):

| form | used by |
| --- | --- |
| Haldane's recombination fraction of one meiosis, (1 - e^(-2d)) / 2 (`haldane_fraction`) | painting's switches between bins (at least 1e-15), family phase and phase polishing, the recombination decoder, its shared-orientation screening and the junction detector's hotspot ceiling |
| the probability of at least one event of a Poisson process, 1 - e^(-d) | pedigree's selector switches (`poisson_switch_stay_terms`, which also returns the log switch and stay costs) and local selection's copying path model, with d multiplied by the number of generations |

Family phase evaluates the Haldane fraction of its marker intervals with NumPy's vectorised `expm1`, which can
differ from the compiled kernels' scalar `expm1` in the last bit.

Assembly's linker and endpoint beam use genetic distances between blocks the same way (the beam's maximum
transition gap comes from the map and the number of generations). The map's knots and fallback rate are part of the
identities of every stage that uses them, so a changed map gives new results in a new output root rather than
reusing old ones.

## What an inferred map means

The input map is a prior for the recombination process, not ground truth. The recombination stage estimates rates
conditional on the final phase and the fixed inferred pedigree ([recombination](methods/recombination.md)).
Informative data can support departures from a positive prior; regions without informative meioses stay
prior-sensitive, and a zero prior forbids crossover mass in its interval. Posterior event mass inside a marker
interval is allocated by the prior's integrated genetic mass, so structure entirely between uninformative markers
is not learned. Expected crossover counts, called crossover intervals and observable meiosis exposure are separate
outputs; unobserved gaps are not reported as measured regions without recombination. Shared-family
orientation-error evidence (`[recombination] shared_family_evidence`, on by default) changes only the recombination
stage's decoding, never the published phase. Neither an inferred map nor family phase feeds back upstream.
