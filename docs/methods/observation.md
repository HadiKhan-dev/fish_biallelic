# Observation pilot

The observation pilot estimates, before relatedness and every discovery, the run's outlier rate: the probability that a
genotype likelihood is replaced by a draw from the site's cohort genotype law. It fits the real discovery block search
on a fixed, hashed selection of pilot blocks of every analysed contig. N is samples, L sites, U pilot units (sample
blocks) and G the rate grid. Every consumer (relatedness, discovery, local selection, assembly) binds the product's
identity and reads its rate. The rate is a strict-fit discordance rate, conditional on the fitted rows, not a property
of the data.

Discovery, local selection and assembly read the genotype likelihoods at face value: there is no evidence scale. A
scale chosen by held-out prediction on the pilot blocks was built and tested in 4.21 and removed by the owner's
decision of 16 November 2026: on simulations with known truth the held-out criterion preferred a scale of 2 where the
truth preferred 1, and a sound criterion (each candidate scale judged on the whole assembly) would cost about
+115-125 % of pipeline runtime.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md),
[record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ob01"></a>
## ob01 Strict fits and the one-rate estimate

Fit every pilot block with all cells at the strict outlier rate, decode each sample's Viterbi path under discovery's
binned switching model, and on the decoded known-state cells of each sample block sum log[(1 - e) P + e B] over the
rate grid (P the decoded state's predictive likelihood, B the cohort law at the site), one unit per sample block. The
run's rate is the maximum of the units' sum, refined between the grid maximum's neighbours by the vertex in log e of
the parabola through the three grid points (clipped to the neighbours' bracket; ties to the smaller rate), at least
0.5 / cells (a numerical regulariser). A maximum on the grid's lower bound is a legitimate near-zero, published
`rate_at_floor`; on the upper bound it is censored (the rate is at least that), published `rate_censored`. The raw grid
maximum is published beside the refined rate, and a two-level fit (a share of units at a second rate) as a diagnostic
only. The rate never feeds back into the strict fits. Without a decoded cell the strict rate stands (`estimated`
false).

Implementation: [science/observation/run.py](../../haplotype_reconstruction/science/observation/run.py),
[science/observation/pilot.py](../../haplotype_reconstruction/science/observation/pilot.py),
[science/observation/estimate.py](../../haplotype_reconstruction/science/observation/estimate.py),
[stages/observation/tasks.py](../../haplotype_reconstruction/stages/observation/tasks.py).

Parameters by name: `strict_outlier_rate`, `grid_low`, `grid_high`, `grid_points`, `pilot_share`,
`minimum_run_blocks`, `maximum_run_blocks`. Values, units and current rationale are in the
[generated parameter reference](../reference/parameters.md).

Cost and scope: the fits are discovery block searches at one rate, independent tasks; O(N L G) per block for the
decode and O(U G) for the estimate.
