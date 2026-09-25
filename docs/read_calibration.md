# Read-model calibration

Read calibration is enabled by default in `simulate`, `reconstruct`,
`astcal` and `tropheops`. Disable it with `--no-read-calibration`,
`[run].read_calibration = false` in TOML, or
`HAPLOTYPES_READ_CALIBRATION=off` for direct workflow invocations.

## Model and information used

For each chromosome, observed REF/ALT allele depths select among three nested
models: shared binomial, pooled sample-specific binomial, and pooled
sample-specific means with shared heterozygote beta-binomial dispersion.
Selection does not force overdispersion when the simpler model predicts better.

For sample s, error e_s and allele-balance b_s give heterozygote ALT probability
q_s = e_s + (1 - 2e_s)b_s. Homozygotes are binomial with ALT probabilities
e_s and 1-e_s. The joint model uses a beta-binomial heterozygote with mean q_s
and fitted chromosome-shared concentration. A larger concentration approaches
the binomial limit. Error and balance effects on logit scales are pooled using
a diagonal empirical-Bayes approximation. Training-data profile curvature
estimates the pooling strength; the joint fit then estimates sample means,
genotype-mixture nuisances and shared concentration together. This is not
exact Bayesian inference or a guarantee of identifiability at low depth.
A sample with no training-fold reads takes the fitted pooled prior mean for
its error/balance effects, not the initialization of the shared model. Its
unidentified genotype mixture remains uniform.

Each sample has an unconstrained three-genotype mixture for fitting and
predictive scoring. These weights are not HWE frequencies and are **never**
applied as downstream genotype priors. Neither pedigree truth, inferred founder
sequences nor simulation-generating parameters enter estimation.

Every eighth marker contributes to histograms; depths above 48 are omitted
from fitting only. Four physical-window folds are defined by
`floor(position / 1,000,000) mod 4`. Folds 0/1 fit parameters, fold 2 selects
the highest predictive likelihood among converged models, and fold 3 supplies
an untouched predictive diagnostic. Exact ties prefer the simpler model.
The selected training fit is applied across the contig, without a full-data
refit. Unlike the preceding shared-binomial implementation, this is **not**
opposite-fold likelihood application: some inference reads also trained the
model. The diagnostic fold is not used to choose parameters or the model.

All markers and reads contribute to the resulting likelihood tensors, including
depths above the fitting cap. Common integer count pairs use an exact lookup;
higher depths use the same model directly. Zero-depth observations remain
exactly uniform across genotypes and retain their separate missingness mask.
Short/uninformative contigs without observations in all four folds, or failed
shared-model fits, use the fixed model (e = 0.02, q = 0.5), with the reason
recorded. Failed complex fits cannot displace a converged simpler fit.

Optimizer bounds, initializations, numerical tolerances and sampling budgets
remain algorithmic choices, not learned biological parameters. Model selection
is predictive, not a significance test or proof of better founder accuracy.

## Pipeline and checkpoints

The observation model is fitted once per observed contig. The same normalized
raw likelihoods feed discovery, both feedback rounds, founder completion,
L1–L4 assembly, painting, pedigree inference and family phase. Recombination
inference consumes the resulting final phase. Sample order is preserved.

The `block_discovery` payload stores `global_probs` and a `read_calibration`
report with model identity, sample-ordered fitted parameters and nuisance
mixtures, convergence information, fold sizes, selection/test scores, selected
model and fallback reason. Nuisance mixtures are recorded for reproducibility,
not used as output genotype priors. The discovery identity distinguishes this
model from old likelihood checkpoints. Dependent products must be recomputed;
existing results are not deleted. Use an isolated root for comparisons.
Version `heldout-joint-read-model-v3` records corrected no-training-read
pooling and compiled fitting kernels. The likelihoods, optimizer bounds,
iteration limits, held-out selection and convergence tolerances are unchanged.

Simulation templates and read generation retain their independent identities.
Calibrated likelihoods are stored in `block_discovery`, not substituted into
the generating `simulated_reads` payload. Compatible templates and generated
reads remain reusable.

## Founder-count refinement

Final assembly now also tries up to two additional founder paths per component,
stopping at the first rejected addition. The incumbent and count-up proposals
receive symmetric fixed-count beam/dual and paired repairs. An optimistic
free-boundary local score can safely rule out an addition before its expensive
repair; a near-boundary proposal still runs normally. Acceptance uses the
existing `K * complexity_cost - 2 * full_site_score`, unchanged observation
mask and existing local alleles. There is no assumed true K or forced sixth row.

This pass runs once after the final executed hierarchy level, not after every
early level or feedback round. Existing progressive refinement is unchanged.
It has separate resumable checkpoints and diagnostics. The API setting
`FounderRefinementConfig(count_max_additions=0)` disables only additions;
the existing `--founder-refinement off` disables founder refinement as a whole.
The separate multi-model discovery candidate-bank experiment is not enabled.

## Evidence and accepted limitations

Representative indexed AD windows in Tropheops (47,565 SNPs, 116 samples) and
Astcal (59,968 SNPs, 290 samples) selected the joint model. Extraction plus
fitting took 14.3 and 26.1 seconds on 18 CPUs. These are bounded-window
timings, not whole-file or pipeline timings, and measure predictive fit,
not real-data biological accuracy.

The joint model estimates parameters and predicts held-out reads better in
controlled read-stressed simulations, but chromosome-long founder errors
regressed in both full-chromosome controls. On seed7004 chr10, wrong called
founder alleles rose 85,658 to 89,122; on seed7019 chr7, 401,952 to 423,703.
Count-up/refit recovered a fifth row in one candidate-bank regional control,
reducing wrong-plus-missing alleles 51,200 to 28,137 while adding 1,623 wrong
calls; it still rejected the sixth row. These are not uniform accuracy gains.

Both changes were promoted on 24 September 2026 with explicit acceptance of
these trade-offs. Historical comparisons remain preserved. See
[validation](validation.md) for controls and denominators.
Site-specific mapping biases, asymmetric homozygote errors, variant
ascertainment and dependent errors remain limitations. Calibration cannot
resolve ancestral phase that descendants do not identify.

Raw AD is required; PL/GL-only inputs do not generally identify the original
read parameters. Default-on pedigree evidence calibration is a separate,
downstream estimator requiring chromosome-level candidate evidence after
painting. Recombination priors, confidence targets and numerical controls
retain their distinct roles as assumptions or operating choices.
