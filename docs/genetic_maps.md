# Input and inferred recombination maps

The fallback is a float-valued **5.0 cM/Mb**, equivalent to `5e-8` Morgans/bp.
Supply `--rate-cm-per-mb` or TOML `[recombination].rate_cm_per_mb` to change it.
A supplied map replaces that rate between its first and last knot on each
named chromosome. Missing chromosomes and tails beyond the knots use the
fallback; the cumulative map remains continuous at both ends.

## Combined input files

The preferred interchange is a headerless whitespace-separated PLINK/Beagle
four-column `.map` file:

```text
chr1 marker1 0.0 1
chr1 marker2 5.0 1000001
chr1 marker3 15.0 2000001
chr2 marker1 0.0 1
chr2 marker2 2.5 1000001
```

Columns are `chromosome marker_id cumulative_cM position_bp`. Marker IDs are
ignored. The cumulative column must be **centimorgans**, not Morgans. Files
using Morgans must be converted first. An all-zero cumulative map explicitly
means zero recombination between knots, not an absent map.

A headed three-column table is also accepted:

```text
chromosome position_bp map_cM
chr1 1 0.0
chr1 1000001 5.0
```

Chromosome names must match input variants exactly (`chr1` and `1` differ).
Use the same physical-coordinate convention/reference assembly as the VCF/BCF;
coordinates are not shifted. Zero can represent a boundary knot. Within each
chromosome, at least two positions must be strictly increasing and cumulative
cM finite and nondecreasing. Blank/comment lines are ignored, but reversed
records and duplicate positions are not silently sorted or repaired.

```bash
python run.py tropheops --config configs/tropheops.toml \
  --recombination-map work/data/cross.map --rate-cm-per-mb 5.0
```

## Python API and separate chromosome files

```python
from haplotype_reconstruction.core.genetic_map import load_genetic_maps

maps = load_genetic_maps("cross.map", default_rate_cm_per_mb=5.0)
chromosome = maps.for_contig("chr1")
distance = chromosome.interval_morgans(500001, 1500001)  # 0.075 Morgans
```

A mapping of exact chromosome name to filename also accepts headed SHAPEIT
tables with columns `position COMBINED_rate(cM/Mb) Genetic_Map(cM)`. The
cumulative column is authoritative; the local-rate column is checked but not
used for integration. `load_genetic_maps(None)` gives scalar-only behavior.

The map is piecewise linear in cumulative distance. Every transition uses
`G(right)-G(left)`, not one endpoint's rate times the interval length. Each
stage retains its own established mapping function and generation/meiosis
multiplier. Equal cumulative knots define a cold interval. Sparse knots imply
interpolation through their entire gap, not an unspecified internal interval.

Map contents and fallback rate are part of affected checkpoint identities.
Painting checkpoints retain the resolved map so downstream scoring can
reconstruct the painting law. This does not merge disconnected phase components.

## What an inferred map means

The input map is a process prior, not ground truth. T12 estimates rates
conditional on the final phase and fixed inferred pedigree. Informative data
can support spatial departures from a positive input prior; unobserved regions
remain prior-sensitive, and a zero prior forbids crossover mass in that interval.
Posterior event mass within a marker interval is allocated using the input
map's integrated genetic mass, so structure entirely between uninformative
markers is not independently learned.

Shared-family orientation-error evidence is on by default; disable it with
`--no-shared-family-evidence`. This affects T12's conditional decoding and does
not overwrite upstream phase. Expected crossover counts, high-confidence
interval calls and observable meiosis exposure are separate outputs. Unknown
gaps are not reported as measured zero-recombination regions.

For simulations, use `--generating-map` and `--generating-rate-cm-per-mb` to set
the generating process independently of the inference settings. Neither an
estimated map nor family refinement is automatically fed back upstream.
