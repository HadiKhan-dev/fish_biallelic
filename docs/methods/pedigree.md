# Pedigree inference

Pedigree inference estimates, for every eligible sample, how many of its parents are among the sequenced candidates
(its parent-count state: M0, M1 or M2 observed parents) and, where the evidence allows, which samples they are. It
works genome-wide and without generation labels, from two kinds of evidence: per chromosome, the fit of each
candidate configuration to the child's raw genotype likelihoods under a transmission model built from every sample's
painting; and, across samples, finite direction and family evidence from painted junction counts and from the
competing configurations of related samples. Chromosome bootstrap and leave-one-chromosome-out refits measure the
stability of the full-data decisions, and two evidence tiers release calls; Tier B is the primary product. A
full-marker Mendelian exclusion can then settle zero- or one-parent counts that Tier B leaves open, without adding
parent edges. The stage (`pedigree`, entry point `pedigree.run`) runs once per run, genome-wide, after every contig
is painted; [family phase](family_phase.md) and [recombination](recombination.md) condition on its released Tier-B
parent edges, and nothing downstream feeds back into it.

## Inputs and outputs

| | record or setting | content |
| --- | --- | --- |
| reads | `painting` (each contig) | the contig's `ChromosomePainting` ([painting](painting.md)): per component the selected sites and trajectory-class alleles, BACKGROUND's allele frequency, the bins (edges, centres), the normalised transition table, the MAP state classes and track statuses, the emission settings, and the packed emissions when the painting kept them; the painting must have been computed from the contig's current evidence record |
| reads | `painting/summary` (each contig) | component shapes (bins, states), for memory planning |
| reads | `evidence` (each contig) | positions, normalised genotype likelihoods `float64[N, L, 3]` and the positive-depth mask `bool[N, L]` of all samples ([evidence](evidence.md)) |
| reads | the contig's genetic map | `[genetic_map] file` or the fallback `[genetic_map] rate_cm_per_mb` ([genetic maps](../genetic_maps.md)): switch probabilities of the transmission selectors |
| reads | `ParentEligibility` | built by the driver from `[pedigree.eligibility]` and the `[samples]` workbook ([parent eligibility](#parent-eligibility)) |
| writes | `pedigree/prepared`, `pedigree/screen`, `pedigree/scores`, `pedigree/exclusion` | per-contig records ([records, identities and resume](#records-identities-and-resume)) |
| writes | `pedigree`, `complete/pedigree` | the `PedigreeProduct` in the run record, and the stage's completion marker |
| publishes | `OUTPUT/pedigree/` | 15 tables and `evidence_calibration.json` ([published tables](#published-tables)) |

The stage reads no metadata and no published file; it needs at least three samples and a painting of every
configured contig. Its user settings are `[pedigree] evidence_calibration` (`"predictive"` by default, or
`"off"`) and the eligibility policy (`[pedigree.eligibility] policy`, `constraints`, `parent_generations`,
`opposite_sex_pairs`) with the `[samples]` workbook; see the [configuration reference](../configuration.md) and
[usage](../usage.md). Every other value below is an internal constant of `PedigreeConfig` (`pedigree/config.py`) or
of a named module, listed in the configuration reference.

Notation: N samples in run order; C informative chromosomes; per painting component, S states (its trajectory classes
plus BACKGROUND, so S ≤ K + 1 for K founder rows), B painting bins and M selected markers; T rows of the M2 candidate
panel; A configuration rows.

## Parent eligibility

`ParentEligibility` (`pedigree/eligibility.py`) is the candidate universe of the inference. It encodes design
assumptions supplied by the caller, never individual parentage, over the run's sample axis:

- `children`, bool [N]: the inference targets. Only targets receive calls; other samples report
  `excluded_by_parent_eligibility`.
- `parents`, bool [N, N] (child, parent): the admissible M1 edges; never the diagonal, never a non-target's row.
- `compatible_pairs`, bool [N, N] symmetric, or none: the pair rule. An unordered pair {a, b} is an admissible M2
  configuration of child c when both of its edges are admissible and the pair is compatible; no N × N × N table is
  formed.
- `chronology`, bool [N, N], a subset of `parents`: edges whose direction the design fixes. Such an edge is exempt
  from the inferred orientation, reciprocal-family and short-ancestry terms (its own side only, never an uncertain
  co-parent); its genetic evidence, exposure gate and release thresholds are unchanged.
- the policy name, its plain-language assumptions and the metadata fields read. With the masks they form the
  eligibility identity, and the product's eligibility summary records them with `individual_parentage_ground_truth`
  false.
- `sexes` (release 2.2): the recorded sex label of every sample from the `[samples]` workbook (upper-cased; `F`,
  `M`, anything else or blank = unknown), or none without metadata (unrestricted and constraints policies). Sexes
  are never inferred. They order released parent pairs (below) and, when present, enter the eligibility identity;
  the summary counts the recorded labels.

The driver builds the record with `pipeline/samples.py` before any computation, so a wrong workbook, sheet, sample
list or label fails at once. `[pedigree.eligibility] policy` selects one of three policies:

- `"none"` (the default): every sample is a target and every other sample a candidate, every pair is admissible and
  nothing is chronology. Simulations always use it: they accept neither a workbook nor an eligibility policy, so
  neither the generating pedigree nor generation labels reach inference.
- `"constraints"`: sample-name constraints from the JSON file named by `[pedigree.eligibility] constraints`, applied
  to the full universe:

      {
        "candidate_parents": {"fish_c": ["fish_a", "fish_b"], "fish_d": []},
        "excluded_samples": ["outside_pedigree"],
        "ineligible_children": [],
        "direction_supported_edges": [["fish_a", "fish_c"]]
      }

  `candidate_parents` replaces a listed child's candidates (an empty list allows no observed parent; unlisted
  children keep every other sample). `excluded_samples` are neither targets nor candidates; outside-pedigree samples,
  such as other-species references, belong here. `ineligible_children` stay candidates but are not targets.
  `direction_supported_edges` lists [parent, child] pairs of independently supported chronology; each must be an
  eligible edge, and none asserts parentage. Unknown keys or sample names are errors; every pair is admissible.
- `"cohorts"`: generation and sex labels from the `[samples]` workbook. `[pedigree.eligibility] parent_generations`
  maps each target generation to its parent generation, e.g. F2 to F1. The targets are the samples of the listed
  child generations; every sample of a target's parent generation is a candidate of that target; every eligible edge
  is chronology (the generation step orders the two cohorts, not individuals); samples of any other generation are
  neither targets nor candidates. With `[pedigree.eligibility] opposite_sex_pairs` (default true) an M2 pair needs
  one sample recorded F and one recorded M (other sex labels pair with nobody; parent order stays arbitrary).

The presets encode the two real crosses with the cohort policy. `tromau`: F2 ← F1 with opposite-sex pairs; the G0
rows enter discovery and painting (they are the reference rows of the reference-consistency report) but, like every
sample not labelled F1 or F2, are neither targets nor candidates. `aulstu`: F2 ← F1 and F3 ← F2 with opposite-sex
pairs; labels starting with G0 read as G0, labels must be G0, F1, F2 or F3 and sexes F or M, and G0 samples are
outside the universe. The labels restrict who may be a parent, not who is: every F1 is an equal candidate of every
F2; no pair of G0 samples is assumed to be anyone's parents (G0 rows are not candidates at all); samples of the other
species never enter the universe; and the calls remain inferences, with no truth labels anywhere. Because every
eligible edge is chronology in these designs, the direction and family terms contribute zero there, and the genetic
evidence, the exposure gate, resampling and the release rules decide.

Eligibility restricts pedigree inference only. Founder discovery and painting use every sample of the input, so a
sample that must not contribute to them has to be left out of the input VCF.

## Preparation: markers, source factors and information blocks

Chromosome workers prepare every painting component of a contig (`pedigree/prepare.py`, `pedigree/sources.py`):

- **Selected markers.** Of each painting bin's selected sites, at most
  `PedigreeConfig.preparation.markers_per_bin = 10` evenly spaced ones (in a bin of n > 10 sites, those at offsets
  ⌊j n / 10⌋, j = 0, …, 9) enter the transmission model, with every sample's genotype likelihoods and observed flags
  there. The likelihoods are used exactly as the evidence stage stored them.
- **Exact source factors.** Within the reduced source model below, each sample's source chain is an HMM over ordered
  diplotypes (S² states) with the painting's own tempered emissions (the packed emissions of the painting record, or the
  same emission model recomputed and divided by its design effect (each sample's bin, as painting does) for components
  beyond the painting's cache) and its normalised label-level transition table at the painting's estimated
  ancestry-change rate, whose weights depend only on whether zero, one or both founders change. This is pedigree's own
  reduced source model, not the marginal of painting's class x fidelity chain: the panel-error state is left out and a
  called founder allele is transmitted unchanged (release 2.14). The backward pass is stored as the bin-0
  ordered-diplotype law (symmetrised over the arbitrary homologue order) and max-normalised right weights r_b, so that
  the posterior chain kernel

      K_b(u, v) = A_b(u, v) r_b(v) / Σ_w A_b(u, w) r_b(w)

  and the per-bin posterior marginals follow in O(S²) per bin through row, column and total sums; the S² × S²
  transition matrix is never formed. A sample without any observed site of the component whose GL row is
  informative (spread above `pedigree.projection.INFORMATIVE_GL_SPREAD = 1e-12`) keeps a uniform law there and is
  evidence-free as a candidate parent of that component.
- **Painted state classes.** The painting's MAP class of each sample, bin and track; BACKGROUND, unanchored tracks
  and abstentions are unknown. They feed the junction counts.
- **Selector switch probabilities.** Between adjacent bin centres at genetic distance d (Morgans, from the contig's
  map, or the fallback rate), θ = 1 − exp(−d), clipped to [1e-15, 0.5]
  (`model.genetic_map.poisson_switch_stay_terms`).
- **Information exponents.** The chromosome's selected markers are grouped bin by bin, across components in order,
  into consecutive groups that close once they hold `PedigreeConfig.preparation.markers_per_information_block = 100`
  markers (the last group may hold fewer). For
  a child with n markers of non-uniform GLs in a bin's group, the bin's tempered-view exponent is η = min(e, n) / n
  with e = `PedigreeConfig.preparation.effective_markers_per_information_block = 3.0` (η = 0 when n = 0): a group of
  linked markers weighs at most about as much as three independent ones.

A component without any non-uniform GL at its selected markers is skipped; a contig without a remaining component is
omitted (`no_observed_nonuniform_component_evidence`). The **informative chromosomes** of the run are the contigs
with at least one prepared component; only they are scored and resampled.

## Ragged quadratic scoring of M0, M1 and M2

The genetic evidence of a configuration is a forward likelihood of the child's genotype likelihoods
(`pedigree/projection.py`, `pedigree/forward.py`). Ragged: every painting component keeps its own state alphabet,
bins and selected markers, with no common grid, and a chromosome's log likelihood of a configuration is the sum over
its prepared components in component order. Quadratic: whatever the configuration, the child's hidden state at a
bin is an ordered pair of transmitted founder states, S² values.

### Parent-count states

M0, M1 and M2 are zero, one and two parents among the eligible sequenced candidates. M0 means zero observed parents,
the parents being unsampled or ineligible; it does not mean a founder or the first generation, and an M0 sample can
have observed offspring. A missing biological parent is never replaced by an unsupported candidate: an M1 call leaves
the second parent unobserved. Each eligible child has one M0 configuration, one M1 configuration per eligible parent
and one M2 configuration per admissible pair of its [candidate panel](#candidate-panel).

### Transmitted-state projection

A parent transmits, at each bin, one of its two painted homologues; which one is a selector that switches at
crossovers with probability θ per bin edge. The exact law of the transmitted founder state couples the parent's
ordered-diplotype chain with the selector, an S⁴ object for two parents. The projection keeps, per candidate and bin:

- π_b(s), the transmitted-state marginal (the homologue-symmetrised source marginals propagated with the selector);
- at each selected marker, the probability that state s transmits the ALT allele given the candidate's own reads:
  the untransmitted homologue is summed out under the source posterior, with the candidate's GL rows mixed with the
  painting's robustness weight (a uniform genotype mixture); a class with a called allele transmits it unchanged,
  and an uncalled class or BACKGROUND starts from BACKGROUND's ALT frequency;
- between adjacent bins, a maximum-entropy bridge: the S × S transition that matches the exact adjacent marginals
  π_b and π_b+1 and the exact same-state (persistence) flow. It is a diagonal plus a rank-one off-diagonal part with
  pivot arms, found by bisection on the maximum-entropy root (at most
  `PedigreeConfig.scoring.bridge_max_iterations = 128` steps, until the bracket stops shrinking), and it is accepted
  only when its rows are stochastic and it reproduces the adjacent marginal and the persistence flow within
  max(8 × `PedigreeConfig.scoring.bridge_tolerance = 2e-13`, 256 ε_mach); otherwise the stage stops with an error.

Where a candidate has no informative markers, its chain between two bins is the painter's interpolation only (the
painting's label-level copying process; before 2.14 an imposed switch-suppression prior). Each bridge is therefore
mixed with the uniform selector kernel,

    T' = (1 − k) T + k U,   k = θ (1 − min(1, n / G)),

with n the smaller number of the candidate's observed informative selected markers in the two adjacent bins and
G = `PedigreeConfig.scoring.source_gap_support_sites = 1`: the mixture acts only next to data-free parent bins and
leaves every data-supported bridge unchanged.

### Child forward model and composite score

An unobserved parent's slot follows the null source: a uniform initial state and, per bin edge, (1 − θ) times the
painting transition's single-homologue marginal plus θ times the uniform kernel. M0 has two null slots; M1 takes the
first transmitted state from the candidate's projection (π at bin 0, then its bridges) and keeps a null second slot;
M2 takes both from the two parents. At an observed selected marker with transmitted ALT probabilities a1 and a2, the
child's emission is

    L(a1, a2) = 3 (g0 p0 + g1 p1 + g2 p2),   p0 = (1 − f1)(1 − f2),   p2 = f1 f2,   p1 = 1 − p0 − p2,
    fi = μ + (1 − 2μ) ai,

with the child's normalised GL row g and μ = `PedigreeConfig.scoring.mismatch_probability = 0.01`. A bin's emission
is the product over the child's observed markers there; unobserved markers contribute 1 (the factor 3 makes a flat
GL row neutral).

Each configuration H is scored by two forward passes over the same markers, sources and transitions: Z_η(H), whose
bin emissions are raised to the child's information exponent η before the sum over transmission paths, and Z_1(H)
at full strength. The chromosome score is

    s(H) = ½ log Z_η(H) + ½ log Z_1(H).

Tempering before path marginalisation can suppress the contradictions that separate a missing parent from a related
observed candidate; the full-strength view keeps them, while the tempered view limits the weight of dense linked
markers. The equal weight is fixed (never fitted per sample or against truth), and s is a composite score over two
evidence weightings of the same data: neither two independent data sets nor a calibrated likelihood. Each view keeps
its own scaling and falls back to log-space products where a product would underflow. A wholly unobserved child is
neutral, and an evidence-free candidate adds nothing: its M1 score equals the child's M0 score, and an M2 row
containing it reduces to the other parent's M1 score (or to M0).

## Candidate panel

M0 and a certified genome-wide M1 profile are computed first. Unproved eligible dyads receive exact scoring; retained identities keep raw scores and excluded identities share the fixed floor profile described below. M2 is scored only for the complete retained-parent panel (`pedigree/panel.py`).
For child c with n_c eligible parents, the robust screen utility of parent p is

    u(c, p) = Σ_k w_k [ρ r_k(p) + (1 − ρ) ((1 − γ) σ_k(p) + γ / n_c)].

On chromosome k, σ_k is the softmax of the child's M1 screen scores, centred at their maximum, divided by the
temperature τ_k = max(1, ⌈m_k / 100⌉)^q and clipped at −60 nats, with m_k the chromosome's selected markers, 100 the
information-block size and q = `PedigreeConfig.panel.information_tempering_power = 0.5`; r_k is tied-rank evidence
((n_c − average rank) / n_c with ranks counted from 0 for the best score, normalised to sum 1, scores within the
contrast tolerance sharing their rank);
ρ = `PedigreeConfig.panel.rank_weight = 0.35` and γ = `PedigreeConfig.panel.chromosome_contamination = 0.02`. The
chromosome weights are √m_k / median, clipped to [1/R, R] with R = `PedigreeConfig.panel.maximum_contig_weight_ratio = 4.0`
and normalised to sum 1, so no chromosome dominates the screen.

The panel holds every admissible pair among each child's genome-retained parents,
including all cutoff ties. It has no fixed top-parent cap. It is built once before
calibration and remains fixed in every resampled fit. The full eligible-pair prior
denominator is retained. Individually weak M1 parents may form a strong raw M2 pair;
excluding such a pair is an explicit model approximation, and M2 evidence remains a
lower bound when admissible pairs are outside the retained panel. Robust utility is
reported for diagnostics; it does not select an arbitrary prefix.

## Exposure and junction counts

The score phase also extracts, per informative chromosome, the two structural summaries the decision uses.

**Exposure** (`pedigree/exposure.py`) comes from the exact source marginals, read as each sample's posterior mass
over unordered founder pairs per bin. A founder class is anchored when the component calls it somewhere; BACKGROUND
and unanchored trajectories are never identity evidence, and evidence-free samples contribute nothing. An edge
(child, parent) is matched in a bin with the posterior probability that the child's pair shares an anchored founder
with the parent's pair, and exposed with the probability that the bin can test the edge: a match, or both pairs fully
anchored and disjoint. A panel pair (child, parent 1, parent 2) is explained when one parent can supply each of the
child's founders, and exposed when explained or when all three pairs are fully anchored and no such assignment
exists. Summed over bins and components these give the expected matched and exposed bins of every eligible edge and
panel pair; the exposure gate of each fit reads the exposed bins, and matched-to-exposed ratios are diagnostics.

**Junction counts** (`pedigree/junctions.py`) are the minimum number of founder switches of each sample's painting
per chromosome, invariant to phase. Within each bin, trajectory classes with identical alleles at the selected markers
are pooled into one local class, so splitting one identical-by-state haplotype into duplicate classes cannot
manufacture switches. The hidden state is an unordered pair of deduplicated whole-component trajectories (possibly
the same one twice), with cost 0 to stay, 1 to change one trajectory and 2 to change both; per-trajectory and global
minima keep each update quadratic in the number of trajectories. Unknown painted classes are uninformative, and the
callable haplotype-bin labels are counted alongside, so that an incomplete painting cannot pass as a shallow one.

## Cross-chromosome evidence calibration

With `[pedigree] evidence_calibration` set to `"predictive"` (`PedigreeConfig.evidence_calibration = "predictive"`),
one positive scale λ multiplies every chromosome's M0, M1 and M2 log likelihoods before the decision
(`pedigree/calibration.py`); `"off"` keeps λ = 1. The scale is chosen by predicting held-out chromosomes. The
informative chromosomes, by index modulo 4, form four folds; each of folds 0 to 2 is predicted by the other two, and
fold 3 is an untouched test. For each eligible child with at least two configurations finite on every chromosome,
the objective sums over the three fitting folds f

    log Σ_r exp(λ ℓ_−f(r) + log π_r + m_f(r) + ℓ_f(r)) − log Σ_r exp(λ ℓ_−f(r) + log π_r + m_f(r)),

the log predictive mixture of fold f's likelihoods ℓ_f under configuration weights from the two other fitting folds
ℓ_−f, tempered by λ. Here π_r is the fixed state prior divided by the configuration's eligible identity count, and
m_f(r) = −∞ excludes reciprocal-dominant rows: a row naming p as a parent of c is left out of the objective when, on
the fitting chromosomes, M1(p with parent c) − M0(p) ≥ M1(c with parent p) − M0(c). Genetics cannot order such
parent–offspring reversals; the direction and family evidence settle them, and decisions still see every row. A grid
of `pedigree.calibration.GRID_POINTS = 33` points over log λ in [log 0.001, log 100] brackets the local maxima (the
objective need not be unimodal), and each is refined by bounded scalar optimisation.

The fitted λ is applied only when the optimiser succeeds, the gain over λ = 1 exceeds
`pedigree.calibration.MINIMUM_GAIN = 1e-08` nats, the optimum lies more than
`pedigree.calibration.BOUNDARY_MARGIN = 0.01` (in log λ) inside the bounds, and λ < 1: the weight only tempers, never
sharpens. Otherwise λ = 1, with the reason as status (`no_predictive_gain`, `above_unit_scale`, `boundary_optimum`,
`optimizer_failed`, `insufficient_alternatives`). With fewer than two informative chromosomes in any fold, that is
with fewer than 8 informative chromosomes, λ = 1 and the status is `insufficient_contigs`. The test fold, predicted
from all three fitting folds, is recorded as `untouched_test_gain` (at the applied scale) and
`fitted_untouched_test_gain` (at the fitted scale) and never selects anything; the whole report is
`evidence_calibration.json`.

λ is a conditional predictive weight of a composite model, not a calibration of posterior probabilities: the
paintings and the candidate panel use all chromosomes, relatives are dependent, and the priors, candidate universe,
direction model and release rules stay fixed. Resampled fits condition on λ (its estimation uncertainty is not
resampled), and the full-marker exclusion uses the raw genotype likelihoods without it.

## One weighted fit

The decision (`pedigree/decision.py`) runs one fit on the full data and one per resampled chromosome set. A fit gives
each informative chromosome k a multiplicity w_k (all 1 for the full data, the draw counts in a bootstrap replicate,
one zero in a leave-one-chromosome-out refit) and recomputes everything that follows the genetic scores
(`pedigree/selection.py`, `pedigree/orientation.py`, `pedigree/family.py`, `pedigree/graph.py`); the scores, λ and
the panel stay fixed.

### Configuration likelihoods and the exposure gate

The genome-wide log likelihood of a configuration H is the weighted chromosome sum ℓ(H) = Σ_k w_k λ s_k(H). Each M1
and M2 likelihood is mixed once with the child's M0 likelihood,

    ℓ'(H) = log((1 − κ) exp ℓ(H) + κ exp ℓ(M0)),

with κ = `PedigreeConfig.selection.m0_contamination_probability = 0.02`: one mixture per genome-wide configuration,
not per chromosome, so a configuration that fails somewhere cannot fall more than log(1/κ) ≈ 3.9 nats below "no
observed parent".

A non-M0 configuration is testable in the fit only when each of its edges, and for M2 its panel pair, has weighted
exposed bins of at least `PedigreeConfig.exposure.minimum_edge_exposed_bins = 1.0` (pairs:
`PedigreeConfig.exposure.minimum_pair_exposed_bins = 1.0`), at least
`PedigreeConfig.exposure.minimum_exposed_fraction = 0.1` of the weighted painting bins of the fit's chromosomes, and
exposure on at least `PedigreeConfig.exposure.minimum_exposed_contigs = 3` distinct chromosomes of positive
multiplicity (a chromosome counts when it alone holds the minimum exposed bins; repeated bootstrap draws count once).
An untestable configuration keeps the child's M0 likelihood as its state score, so it dilutes its state's evidence
towards M0, and its identity cannot be named (identity score −∞). This hard gate sets the
[chromosome-count requirement](#chromosome-count-requirement). Matched-to-exposed ratios against the thresholds
`PedigreeConfig.exposure.report_edge_coverage = 0.95` and `PedigreeConfig.exposure.report_pair_explainability = 0.95`
appear in the diagnostics only and never gate a decision.

### Finite continuous direction

Junction counts orient a linked pair. A homologue transmitted by parent p copies one of p's homologues at each
position, so on chromosome k it carries each of p's J_p painted junctions with probability 1/2, seen where the child
is callable (fraction f_c of its haplotype bins, missing at random). Everything else in the child's count J_c (the
crossovers of the two meioses that formed the child, junctions from the other parent and false junctions) is a
non-negative remainder with no assumed distribution:

    L_k(p → c) ∝ P(Binomial(J_p, f_c / 2) ≤ J_c),     L_k(c → p) ∝ P(Binomial(J_c, f_p / 2) ≤ J_p).

Under the reverse orientation the proposed parent would have lost junctions that its own "parent" transmits: equal
counts give no evidence, and a junction-poor parent of a junction-rich child gives strong evidence. With probability
ε_c = `PedigreeConfig.direction.chromosome_error_probability = 0.05` one chromosome's comparison is painting failure,

    π_k = (1 − ε_c) L_k(p → c) / (L_k(p → c) + L_k(c → p)) + ε_c / 2,

so each chromosome contributes at most log(2/ε_c − 1) ≈ 3.7 nats, and the log odds add over chromosomes as
independent meioses, LLR = Σ_k w_k logit π_k. Painting failure need not be independent between chromosomes (a
sample-level junction excess or deficit repeats on every chromosome), so with probability
ε = `PedigreeConfig.direction.pair_failure_probability = 0.02` the pair's whole comparison carries no orientation
evidence, and each proposed parent of a configuration adds the neutral-centred correction

    log(2 ((1 − ε) σ(LLR) + ε / 2)),

which is zero for equal support and lies within [log ε, log(2 − ε)] ≈ [−3.9, +0.7] nats. The expected number of the
child's own crossovers is deliberately not imposed (it depends on the recombination rate, on whether the transmitting
parent's homologues differ at a crossover, and on painting sensitivity), so related samples of equal junction burden,
such as full sibs or a backcross offspring of a deep parent, receive little orientation evidence. Chronology edges
take zero. A separate paired junction-density contrast, an order probability in [0, 1] (0.5 = no information), only
orders the graph search and is reported as the role probability; it never enters a score.

### Reciprocal family messages

For every sample the competing M0, M1 and M2 configurations are retained. Their weights in the family messages are
the identity scores after the exposure gate and the orientation correction, each divided by its state's eligible
identity count (equal M0/M1/M2 weight; the configured state priors act only in the state choice). A configuration
naming p as a parent of c receives the message

    log P(p does not name c as its parent),

taken over p's configurations with the reverse message, from c to p, excluded: a cavity distribution, so a sample
never receives its own evidence back as support. The message compares p's configurations that do not make c its
parent with all of p's configurations, marginalising over p's possible co-parents rather than freezing a selected
pedigree. `PedigreeConfig.direction.message_passes = 4` synchronous passes, each excluding the incoming reverse
message before forming the next, prepare the short-ancestry path factor. Included and excluded masses are carried as
log odds until the reverse message is excluded, so very small probabilities are not rounded away before the cavity
step.

### Joint short-ancestry correction

Direct reciprocity does not catch a proposed parent who is actually a grandchild, which matters after related
individuals mate: a double grandchild can resemble a direct parent. For a configuration of child c with parental set
P, the correction averages, over the joint configurations of the members of P (each conditioned on not choosing c
directly, which reciprocity already handles), the compatibility Π_g P(g does not choose c as its parent) over the
distinct intermediate ancestors g in those configurations, with configuration weights from the identity scores plus
the preparatory reciprocal term and the outgoing cavity messages of the four passes.
Members of P are excluded as intermediates (a reverse path through the co-parent is already forbidden by
reciprocity) and shared intermediates count once; together these keep legitimate backcrosses. Only the top
`PedigreeConfig.direction.ancestry_path_budget = 16` configurations of each parent (by weight) are enumerated; all
omitted mass, including ties at the panel boundary, counts as compatible, so the truncation can only weaken the
penalty and never depends on sample order. The term min(0, log averaged compatibility) is added to both the state and
the identity score of the configuration. Chronology exempts its focal side: with one exempt side only the uncertain
co-parent's paths count (both focal parents stay excluded as intermediates). The term is finite evidence, not a veto
from a called pedigree, and it is never fed back into the messages it was computed from.

### Converged final reciprocal solve

With the path factor frozen, the preparatory reciprocal term is replaced by a converged solve of the same cavity
equations on the path-adjusted scores: synchronous undamped iterations until the largest change of a log message is
at most `PedigreeConfig.direction.final_tolerance = 1e-08`, for at most
`PedigreeConfig.direction.final_max_iterations = 8192` iterations (slow monotone modes need thousands, and damping
would halve their rate); on failure, a restart of the same equations with damping
`PedigreeConfig.direction.retry_damping = 0.5` for at most `PedigreeConfig.direction.retry_iterations = 4096`
iterations. The residual is measured before damping, so smaller damped steps cannot signal convergence falsely.
Failure of both attempts is an error in the full-data fit and an abstention, for every sample, of a resampled fit
(which stays in the denominator and so only lowers support). The incoming final messages are added to the state and
identity scores once, and the path factor is not recomputed from them. The product records the full-data solve
(iterations, residual, retry use) and the counts of non-converged bootstrap and LOCO fits as `family_solve`.
Excluding immediate feedback does not exclude every loop: the messages are a bounded, loopy approximation, not exact
pedigree marginalisation.

### Marginal state, then identity

The state evidence of state m for child c is the log mean likelihood of its configurations,

    E_m(c) = log Σ_{H in m} exp v(H) − log n_m(c),

with v(H) the configuration's state score (likelihood after the M0 mixture and the exposure gate, plus the
orientation, path and final reciprocal terms) and n_m(c) the full eligible identity count (1 for M0, the eligible parents for M1, the admissible pairs for M2), so
an incomplete M2 panel lowers the M2 evidence and never inflates it. The marginal state score adds the log of the
fixed state prior, 1/3 for each state in the decision fit, and the state support is the softmax of the marginal
scores. The state is chosen first, as the unique winner of the marginal state scores, then the identity within it,
as the unique winner of the decision scores (marginal state score + identity score − the state's best identity
score). Two values closer than the contrast tolerance, `pedigree.ties.CONTRAST_ULP_FACTOR = 512.0` machine epsilons
times the largest magnitude compared (taken as at least 1), are tied: a tie leaves the state
(`no_unique_marginal_parent_state`) or the identity (`parent_state_resolved_identity_unresolved`) unresolved, so
floating evaluation order never decides a call.

M0 calls must survive prior sensitivity: the selection is repeated on the same adjusted scores under each
alternative fixed prior, (0.5, 0.3, 0.2), (1/3, 1/3, 1/3) and (0.2, 0.3, 0.5) for (M0, M1, M2), and an M0 state is
released only when every prior also selects M0 (the exact M0 configuration additionally needs an M0 graph choice
under every prior).

### Acyclic graph selection

The local winners are kept when their parent edges form a directed acyclic graph. Otherwise cycles are examined
first: in each cyclic strongly connected component of the local edges, the implicated child with the weakest support
(confidence = the smaller of its state and identity margins) is the link the search is expected to break, and when
two or more implicated children are tied at that weakest link all of them are marked as graph tie conflicts and left
unresolved, because choosing one by sample order would manufacture confidence from a graph constraint. Peeling
repeats until no cycle remains. The tie tolerance of this step is 512 machine epsilons times the largest magnitude
of the scores the implicated children's margins were computed from (their marginal state scores and the identity
and decision scores of their local state; at least the margins themselves and 1), the same yardstick as a single
margin's tie test above: a margin is a difference of composite scores of 1e4-1e5 nats and carries their rounding
(measured: up to about 136 ULP of its scores, so a difference of two margins up to about 272).

A greedy pass then visits the children that have a local configuration, tie conflicts excluded, by descending
orientation support of that configuration (the product of its edges' order probabilities), then confidence, then
sample index. A child keeps its local
configuration when it fits the graph; otherwise it takes the unique winner, by decision score, of the highest
feasible state at or below its local state (M2, then M1, then M0), never a higher parent count, so the graph
constraint cannot create parent evidence. Up to `PedigreeConfig.selection.graph_local_search_passes = 3` coordinate
passes then re-select each child given the others, stopping early when nothing changes. Displaced children are
reported (`DAGDisplaced`, `GraphConflict`).

## Chromosome resampling

Support comes from refits with resampled chromosome multiplicities (`pedigree/resampling.py`):
`PedigreeConfig.resampling.bootstrap_replicates = 1000` bootstrap replicates, each drawing C informative chromosomes
with replacement from a generator seeded with `PedigreeConfig.resampling.bootstrap_seed = 20260725`, and C
leave-one-chromosome-out (LOCO) refits, counted separately. Every refit recomputes the orientation, the exposure
gate, the reciprocal and path terms, the final solve, state, identity and graph; the genetic scores, λ, the panel,
the eligibility, the priors and the thresholds stay fixed. Counts are integer sums over fits, per configuration (as
local winner and as graph choice), per state and per (child, parent) edge, and a fraction divides by all fits,
including non-converged ones that abstained. The fractions measure internal stability across chromosomes, not
calibrated probabilities of correct parentage, and they condition on the paintings, the panel and λ, which use all
chromosomes.

## Release tiers

`pedigree/release.py` turns the full-data fit and its support into two evidence tiers. Thresholds label evidence
tiers; they are not biological acceptance probabilities.

| tier | state and configuration bootstrap | parent bootstrap | LOCO |
| --- | --- | --- | --- |
| Tier B (primary) | `PedigreeConfig.release.tier_b.configuration_bootstrap = 0.7` | `PedigreeConfig.release.tier_b.parent_bootstrap = 0.7` | `PedigreeConfig.release.tier_b.loco_fraction = 0.8` |
| Tier A (strict) | `PedigreeConfig.release.tier_a.configuration_bootstrap = 0.95` | `PedigreeConfig.release.tier_a.parent_bootstrap = 0.95` | `PedigreeConfig.release.tier_a.loco_fraction = 0.95` |

### Tier rules

A tier releases a child's parent state when the full-data fit resolved the state, at least
`PedigreeConfig.release.minimum_informative_contigs = 3` chromosomes are informative for the child (its own
per-chromosome state evidence has a unique winner there), the state's bootstrap and LOCO fractions reach the
tier's state bootstrap and LOCO cut-offs, and an M0 state survives every alternative prior. It releases the exact
configuration when, in addition, the local configuration is also the graph's choice without a tie conflict, the
configuration's bootstrap and LOCO fractions as local winner and as graph choice reach the cut-offs, an M0
configuration is the graph choice under every prior, every named parent is stable (parent bootstrap and LOCO
fractions at their cut-offs), and every inferred parent edge is decisively oriented (below). A released state
without its configuration states why, as `<tier>_parent_state_supported_<reason>` with the reason
`graph_tie_conflict`, `graph_conflict`, `orientation_unresolved` or `identity_unresolved`; an unreleased state is
`unresolved_below_<tier>_state_support` or `unresolved_<tier>_m0_prior_sensitivity`.

### Orientation of released edges

Chromosome resampling measures stability, not direction: two strongly linked samples whose junction counts cannot
orient them can be resampled into either of two nearly equal reciprocal configurations and pass the support cut-offs
in the wrong direction. A released inferred edge p → c therefore also needs a decisive full-data orientation. With
every other converged reciprocal message held fixed, the pair's anti-reciprocity factor is evaluated exactly: for
the cavity inclusion log odds a (of c's configurations naming p) and b (of p's configurations naming c), the joint
states p → c, c → p and neither weigh e^a : e^b : 1, and the edge is released only when

    a − b ≥ log((1 − q) / q) ≈ 3.9 nats,

with q = `PedigreeConfig.release.orientation_reversal_probability = 0.02`. Otherwise the state stays released with
status `<tier>_parent_state_supported_orientation_unresolved` and the edge is withheld from the partial and
candidate-set views, so family phase never conditions on it. Chronology edges are exempt.

### Marginal edges, candidate sets and the complete view

Where a tier does not release the exact configuration, it can still release single parent edges of the full-data
graph (the partial view). For a child without graph conflict and with enough informative chromosomes, a parent that
is the same in the local and the graph configuration is released when its local and graph parent bootstrap fractions
reach the parent cut-off, both its LOCO fractions the LOCO cut-off, and its orientation is decisive (status
`<tier>_marginal_parent_support`, or `<tier>_marginal_parent_orientation_unresolved` when only the orientation
fails). The partial view states the observed-parent count as bounds: the released count when the state is
released, otherwise from the number of released edges up to 2. An edge does not say whether the child's other parent
was sampled.

Tier-B candidate sets describe what remains open. Within the called state they hold the fewest configurations, by
descending softmax of the decision scores (ties by row), that carry `PedigreeConfig.release.support_set_coverage = 0.95`
of the mass. For M1 they list the candidate parents. For M2 one pair gives its two sides; pairs that all share one
parent make that parent one side and the other parents the other side; pairs without a common parent give no sides,
only the configuration list. A singleton side becomes a named parent when it is stable at the Tier-B parent cut-offs
and decisively oriented; retained marginal edges replace sides, and no complete configuration is invented.

The complete view is the full-data acyclic selection itself: a best guess for every target, displaced and tied
children included, and never a release.

## Chromosome-count requirement

The exposure gate needs at least `PedigreeConfig.exposure.minimum_exposed_contigs = 3` distinct exposed chromosomes
in a fit. In a bootstrap replicate that draws fewer distinct chromosomes, or a LOCO refit that leaves fewer, every
non-M0 configuration is untestable, the state evidence cannot separate M0 from M1, and nothing is called, not even
M0. The support of any call (for a target with at least one eligible candidate) is therefore at most the fraction of
replicates that draw at least three distinct chromosomes, 0.22, 0.66, 0.90 and 0.98 in expectation for C = 3, 4, 5
and 6 informative chromosomes, and its LOCO support is 0 unless C − 1 ≥ 3. With the cut-offs above, Tier B can
release nothing with fewer than 5 informative chromosomes and Tier A nothing with fewer than 6
(`pedigree.tier_b_minimum_contigs` derives the Tier-B count from the configuration by exact combinatorics).

The run reacts in three places. Before any computation, the driver stops a genome-wide run configured with fewer than
3 contigs and warns, when fewer than 5 are configured, that no Tier-B parent can be released, so that family phase
and recombination will have no parent edges. After preparation, the stage stops when fewer than 3 chromosomes are
informative. At the end, the product's `TierReach` records the informative-chromosome count, the largest bootstrap
support this run's draws allow, the largest LOCO support, whether Tier A and Tier B are reachable and the minimum
chromosome counts of both; its one-line statement is printed to the log and published in `OUTPUT/pedigree/summary.json`
(`reach`). Omitted contigs lower the informative
count, so the final statement, not the configured contig count, is authoritative.

## Full-marker Mendelian exclusion of M0 and M1 counts

After the tiers, every eligible candidate of each target child whose Tier-B count is unresolved (no state, or fewer
released parents than the called count) is checked against the full-marker raw genotype likelihoods of the evidence
record, on every configured contig (`pedigree/exclusion.py`). At each site observed in both samples of a dyad, with
normalised GL rows x and y,

    A = (x0 y2 + x2 y0) / 2        the two incompatible (opposite-homozygote) genotype pairs
    C = the largest likelihood among the seven Mendelian-compatible genotype pairs
    U = max(x) max(y)
    D = (1 − δ) C + δ U,

with δ = `PedigreeConfig.exclusion.genotype_replacement_probability = 0.01`. D bounds every compatible read model
with at most a δ share of arbitrary genotype replacement, so A / D has null expectation at most one under calibrated
read likelihoods. The evidence E averages products of convex bets 1 − b + b A / D over the bets
`PedigreeConfig.exclusion.bet_sizes = (0.0009765625, 0.00390625, 0.015625, 0.0625, 0.25, 1.0)` and over the whole
chromosome plus half-overlapping windows of widths
`PedigreeConfig.exclusion.window_widths_bp = (1000000, 5000000, 20000000)` bp; sites missing in either sample
contribute 1, and the genome-wide E is the mean over chromosomes, never a maximum over selected windows. A dyad is
excluded when

    log E ≥ log(2N / α),

with α = `PedigreeConfig.exclusion.alpha = 0.01`. The bound is conditional on calibrated likelihoods and on read
errors independent across markers, not on independent inheritance of SNPs; the allowance 2N covers the at most 2N true
observed parent edges, including when the targets were selected using the same data. Mapping errors, ascertainment
and correlated errors violate these assumptions.

The default scorer can stop likelihood work once completed-window products certify this same
final genome-average threshold. An unfinished product cannot certify exclusion: later factors
can reduce it. The certificate accounts for the number of configured chromosomes and samples;
unproved dyads complete the original calculation exactly. Observed-site counts still cover the
whole chromosome. The exact reference remains available in `pedigree/exclusion.py`.

Certified results carry lower and upper log-evidence endpoints. In `exclusion_evidence.csv`,
`LogE` is the exact value when known and NaN otherwise; `LogELower`, `LogEUpper` and
`LogEExact` expose the interval. `exclusion_diagnostics.csv` adds the corresponding
`WeakestExclusionLogELower`, `WeakestExclusionLogEUpper` and `WeakestExclusionLogEExact`
fields. These columns appear when interval evidence is supplied; zero-target tables keep
their original schema. Bounds never masquerade as exact point evidence. The release decision
and its assumptions are unchanged.

Mendelian compatibility is symmetric and cannot tell parent from child. A compatible candidate of which the child is
already the supported Tier-B parent is ruled out by acyclicity, so the release is conditional on the supported
directions. A child's count is released when every eligible candidate was checked, the child is a target with at
least 3 informative chromosomes and no graph conflict, and the surviving candidates, at most one, equal both its
supported Tier-B edges and its complete-view parents, with the complete-view state equal to their number. The result
is M0 or M1, with status `tier_b_fixed_genome_directed_exclusion` and release evidence
`mendelian_exclusion_and_supported_direction`. No edge is added: an M1 release names an already supported edge and
rules out a second observed parent. The bootstrap and LOCO fractions and `TierBStateCall` keep their resampling
values, `TierBFinalStateCall` records the final call, and the exclusion evidence never enters the scores or the
resampled fits. The release depends on the supported directions being correct and is not an unconditional accuracy
guarantee; M0 still means no observed eligible parent, not founder generation.

## Products and published tables

`PedigreeProduct` (run record `pedigree`) holds the sample axis; the configured and informative contigs, with the
reason for each omitted contig; three relationship views as `PedigreeCalls` (per sample a state 0, 1, 2 or −1 for
unresolved, two parent slots and a status): `calls` (Tier B after the exclusion release; simulation evaluation reads
these), `tier_a` and `complete`; the released edges `edges`; `reach` (`TierReach`); the calibration report,
`family_solve`, the exclusion summary and the eligibility summary; the candidate panel (its rows and the screen
utilities); and the report tables.

### Released parent order (release 2.2)

Scoring treats a parent pair as unordered. Only the released products order it (`pedigree/slots.py`), applied once
to `calls`, `tier_a`, `complete`, `edges` and the relationship and candidate-set tables:

- **dam_sire**: both parents have recorded sexes exactly {F, M}: Parent1 / slot 0 is the dam (F), Parent2 / slot 1
  the sire (M). Family phase gives the child's homologue 0 to slot 0, so the exported GT of such a child reads
  maternal | paternal. This basis needs recorded sexes {F, M} from the configured metadata; nothing is inferred.
- **unordered**: otherwise (no metadata, unknown or equal sexes) the slots carry no biological meaning and their
  order is a canonical label that does not depend on sample names or input order: the parent with more observed
  sites (read depth > 0) first, compared over the configured contigs in order until the counts differ; only parents
  with identical counts on every contig fall back to the name. Under `HAPLOTYPES_TIE_REVERSE=1` this label flips.
- **single**: one observed parent: slot 1 when its recorded sex is M (the sire), else slot 0.

Tables carry `ParentSlotBasis` (`dam_sire`, `unordered`, `single`, `none`) next to Parent2; per-slot columns
(Parent1Supported, Parent1Candidates, …) move with their parent. The diagnostic tables (`diagnostics.csv`,
`call_explanations.csv`, `alternative_explanations.csv`) keep the internal configuration order (sample-axis order).
Simulation evaluation compares parents as sets. 2.1 ordered every pair by sample name, so renaming samples changed
Parent1/Parent2 and the GT order of two-parent children.

### ParentEdges

`ParentEdges` is the only pedigree input of [family phase](family_phase.md) and [recombination](recombination.md);
`pedigree.records.load_edges` decodes it without the report tables. It lists the supported Tier-B edges after the exclusion release
as parent, child and slot arrays (slot 0 for the child's Parent1 column, 1 for Parent2, in the released parent
order above), ordered by child and slot,
with each sample's observed-parent count bounds (minimum = released edges, maximum = 2 unless the count is resolved).
Unreleased candidates, orientation-withheld edges and Tier-A or complete-view information never appear, and an edge
does not establish whether the child's other parent was sampled. Its identity, a digest of the sample axis and the
edge list, is what downstream conditioning binds.

### Published tables

`OUTPUT/pedigree/` holds one CSV per report table, `evidence_calibration.json` and `summary.json`. Parent order within a
configuration follows the released parent order (`ParentSlotBasis`; dam/sire only where sexes are recorded); state
names are `zero_observed_parents`, `one_observed_parent`, `two_observed_parents` and `unresolved`.

| file | one row per | how to read it |
| --- | --- | --- |
| `tier_b_relationships.csv` | sample | the primary calls: Parent1 and Parent2 only where the exact configuration is released (or by the exclusion release), ParentState, ObservedParentCount, InferenceStatus, and ReleaseEvidence (`chromosome_resampling` or `mendelian_exclusion_and_supported_direction`) |
| `tier_b_partial_relationships.csv` | sample | the edge view that forms `ParentEdges`: the released parent of each slot with Parent1Supported and Parent2Supported, ExactConfigurationResolved, ObservedParentCountMinimum and ObservedParentCountMaximum, ReleaseEvidence |
| `tier_a_relationships.csv`, `tier_a_partial_relationships.csv` | sample | the same views at the Tier-A cut-offs (no exclusion release) |
| `complete_relationships.csv` | sample | the full-data acyclic selection, a best guess and never a release (statuses such as `selected_<state>`, `graph_displaced_hypothesis_not_tier_eligible`, `graph_tie_conflict_parent_identity_unresolved`) |
| `tier_b_candidate_sets.csv` | sample | what Tier B leaves open: Parent1Candidates, Parent2Candidates, ConfigurationCandidates, named parents, count bounds and the status (`tier_b_exact_configuration`, `tier_b_partial_parent_with_candidate_set`, `tier_b_candidate_set_identity_unresolved`, `tier_b_parent_state_only`, …) |
| `parent_state_calls.csv` | sample | local and selected states and counts, graph displacement and conflicts, role probability, TierAStateCall and TierBStateCall (resampling), TierBFinalStateCall and TierBReleaseEvidence (after the exclusion release) |
| `diagnostics.csv` | sample | everything behind a call: local, best-M1, best-M2 and complete-view parents; state and identity margins; StateLogEvidence0–2 and StateSupport0–2; bootstrap and LOCO fractions of the local state, configurations and parents; exposure and coverage diagnostics; junction burden and callability; informative chromosomes; prior-sensitivity agreement; tier flags; M2 screen completeness; support sets |
| `call_explanations.csv` | sample | one readable row per sample: the candidate-set columns with the local and selected states, local parents, graph conflict, margins, the local state's bootstrap and LOCO fractions, the M2 search completeness and a plain-language interpretation |
| `alternative_explanations.csv` | configuration | the local and graph configurations and the three leading identities within each state, with separate GeneticLogEvidence (scaled, after the M0 mixture), StructuralStateLogEvidence, StructuralIdentityEligible (the gate), DirectionLogContribution, ReciprocalFamilyLogContribution, AncestryPathLogContribution, adjusted state and identity evidence, ParentStateLogPrior, LogEligibleIdentityMultiplicity, MarginalStateLogEvidence and GraphDecisionUtility; priors, multiplicities, marginal evidence and graph utility are not additive terms of one posterior |
| `search_diagnostics.csv` | sample | the M2 screen: eligible parents, scored pairs and panel parents, ScreenBoundaryRank, ScreenBoundaryGap (weakest panel utility minus the strongest omitted one), omitted parents, the called parents' screen ranks and SelectedParentAtScreenBoundary; M1 always scores every eligible parent |
| `exclusion_evidence.csv` | target dyad | Sample, CandidateParent, LogE (exact genome-wide log E, otherwise NaN), ObservedSites; interval evidence adds LogELower, LogEUpper, LogEExact |
| `exclusion_diagnostics.csv` | target child | compatible, supported-reverse and remaining candidates, the weakest exclusion log E against ThresholdLogE, CompleteCandidateEvidence, Released, ReleaseConditionalOnSupportedDirections; interval evidence adds WeakestExclusionLogELower/Upper/Exact |
| `evidence_summary.csv` | informative chromosome | selected markers, mean minimum junctions and mean callable haplotype bins |
| `chromosome_summary.csv` | configured contig | ComponentCount, ScoredComponentCount, InformativeMarkers, OmittedReason |
| `evidence_calibration.json` | (one object) | the calibration report: mode, applied and fitted scale, status, folds, search grid, selection gain, untouched test gains |
| `summary.json` | (one object) | the informative contigs, the omitted ones with their reasons, the chromosome-count statement (`reach`: whether Tier A and Tier B were reachable with these chromosomes, and why), the final family solve, the exclusion summary and the eligibility summary |

All values are composite log evidence or resampling fractions of the fitted model; none is a calibrated probability,
a proof of biological direction or a validation against truth.

## Implementation and cost

### Parallel structure

Chromosome workers prepare the contigs in parallel, the largest evidence first. The screen and score phases run in
persistent chromosome workers that keep each component's source pair masses, projection and child-side terms resident
across the genome-wide panel barrier: the screen returns only each chromosome's summed M1 matrix and marker count, the
coordinating process builds the panel, and the same workers then score M2, exposure and junctions. The decision runs
in the coordinating process; the resampled fits run in a process pool of single-threaded workers that share the
read-only decision inputs and take chunks of bootstrap replicates and the LOCO refits as tasks. The exclusion uses
chromosome workers again. Inside a worker the kernels run in parallel over samples, edges or panel rows. Parallel
work is admitted by byte estimates computed from the records (the component shapes of `painting/summary` and the
decoded sizes of the evidence, painting and prepared records), never from published files
([performance](../performance.md)).

### Cost in K, N and chromosomes

Per component, with S ≤ K + 1: source factors and marginals cost O(N B S²) (contracted transitions, never S⁴); the
projection O(N M S²) for the transmitted alleles, O(N B S²) for the flows and O(N B (S² + 128 S)) for the bridges;
the forward pass O((M + B) S²) per child and configuration; exposure O((E + T) B S² / 2) for E eligible unordered
edges and T panel rows; junction counts O(N B J²) for J ≤ K distinct trajectories. Nothing in the stage is
exponential in K, no kernel is worse than quadratic in S, and there is no exact-enumeration branch.

Per chromosome the dominant term is the M1 screen, O(N n (M + B) S²) for n eligible parents per child, quadratic in
samples without restrictions; the panel bounds M2 at 190 rows per child, O(190 N (M + B) S²). The decision has
A = O(N²) configuration rows (N M0 rows, one M1 row per eligible edge, at most 190 N M2 rows). One fit costs
O(C (A + N²)) for the weighted likelihoods, exposure and junction log odds (the binomial tails, O(C N² m) for a
typical per-chromosome junction count m, are computed once and reweighted), O((4 + I)(A + N²)) for the messages with
I ≤ 8192 + 4096 final iterations, and O(R A + R N² + R² A₂) for the path factor with R = 16 and A₂ = O(N) M2 rows;
resampling repeats the fit 1000 + C times. The exclusion costs O(D L) for D target dyads and L markers (D at most the
targets times N, quadratic in samples in the worst case), with exact reuse of repeated GL triples up to
`pedigree.exclusion.MAX_PATTERNS = 1024` distinct triples per sample (beyond it a direct per-site scorer gives the
same values).

### Memory

A preparation worker holds the contig's decoded evidence and painting and, per component, the emissions, source
factors and workspace. A scoring worker holds the prepared record, whose largest array is the source right weights
(float64 N × (B − 1) × S²), and per component the unordered source pair masses (N × B × S(S + 1)/2, half the ordered
marginals, which the projection consumes in place), the projection (O(N (M + B) S)) and the child terms. The shared
emission products of called founders are cached in child batches of at most
`pedigree.forward.COMMON_EMISSION_CACHE_BYTES = 268435456` bytes (256 MiB); a batch that does not fit is scored on
the uncached path with identical values. Memory per chromosome worker is therefore O(N B S²). The decision holds
O(C (A + N²)) values, and an exclusion worker one contig's evidence and the dyad results.

### Measured on the reference runs

On the whole-genome reference runs (22 contigs, a dedicated 112-CPU node;
[performance](../performance.md#reference-timings)) the stage took 553 s of the 2:03:41 seed-10000 simulation and
76 s of the 41:10 TroMau run. The simulation has 320 samples without eligibility restrictions (319 candidate parents
and 50,721 candidate parent pairs per child); TroMau's eligibility leaves 96 children with 16 candidate parents and
64 candidate pairs each, so the screen, linear in the candidate parents per child, and the decision, with one M1 row
per eligible edge, are far smaller there. The run record times the stage as a whole, not its phases. Memory per task
is the largest peak one task added to its worker (resident memory, shared pages it touched included):

| phase | simulation seed 10000 | TroMau |
| --- | --- | --- |
| preparation, per contig | 17.9 GiB | 1.4 GiB |
| screen, per contig | 10.9 GiB | 1.2 GiB |
| score, per contig | 0.6 GiB | 0.3 GiB |
| resampling, first task of a worker / later tasks | 0.17 / 0.03 GiB | 0.07 / < 0.01 GiB |
| exclusion, per contig | not run (no candidate dyads) | 0.66 GiB (48 candidate dyads) |

No pedigree task waited for memory, and the preparation, screen and exclusion peaks were at most 0.80 of their
estimates. The node held at most 84 GiB during the stage on the simulation and 28 GiB on TroMau (30-s samples),
against run peaks of 259 GiB (family phase) and 96 GiB (contig lanes). The evidence calibration took 4.1 and 0.5 s,
and the final family solve converged in 4 and 2 iterations with no non-converged bootstrap or LOCO replicate.

### Records, identities and resume

Per contig, each record bound to its identity:

- `pedigree/prepared`: the prepared components (selected-marker alleles and GLs, bin sizes, transition, source
  factors, state classes, switch probabilities, information exponents); bound to the contig's painting and evidence
  record identities, `PedigreeConfig.preparation` and the contig's genetic map.
- `pedigree/screen`: per component M0 [N] and M1 [N, N]; bound to the prepared record, `PedigreeConfig.scoring` and
  the eligibility identity.
- `pedigree/scores`: the chromosome's M0, M1 and panel M2 scores, exposure and junction counts; bound to the screen
  record and the panel's rows.
- `pedigree/exclusion`: the contig's exclusion evidence for the target dyads, written only when there are targets;
  bound to the evidence record, target dyads and `PedigreeConfig.exclusion`; the default certificate also binds its model identity, chromosome count and sample count, so a bound cannot be reused with a different genome divisor or threshold.

The run record `pedigree` (the product) is bound to every contig's prepared and scores identities, the eligibility,
the panel and the whole `PedigreeConfig`; `complete/pedigree` marks the stage done. Every identity also contains the
stage's schema version and code identity (the `pedigree` and `painting` packages with `model/` and `maths/`). Each
record has one current version: a record whose identity matches is reused, and a mismatch stops the run. An
interrupted run therefore resumes per contig (the score phase rebuilds its resident state from the prepared record
when needed), but a changed pedigree setting (including `[pedigree] evidence_calibration` or the eligibility), a
changed painting or evidence record, or changed pedigree or painting code needs a separate output root.

### Determinism

Components are summed in component order and chromosomes stacked in contig order; the panel breaks ties by sample
index; the bootstrap is seeded and support counts are integer sums independent of worker count and completion order;
each kernel task is independent of the thread count. Results do not depend on `[execution] cores`, the number of
chromosome workers or memory pacing.

Convention reversal (`HAPLOTYPES_TIE_REVERSE=1`, release 2.2) reverses the remaining arbitrary orders: the run's
sample axis (centrally), the graph's visit order among equal role and confidence (the index, last key), the search
diagnostics' rank ties, the calibration's choice between equally near scales after the stated prior (smallest
|log scale|), and the projection bridges' lowest state among rounding-tied Hall states, bracketing large branches and
pivots. Not reversed (stated priors and abstentions): abstention on a tie (winners, margins), fewer parents at graph
ties, the symmetric top-16 boundary, and the released parent order's dam/sire basis.

## Interpretation and limitations

- Calls are inferences of a composite model. M0 means no observed eligible parent, not founder generation; an M1 call
  leaves the other parent unobserved; the order of two parents is arbitrary.
- Tier B is the primary product and the only one used downstream; Tier A is a stricter view and the complete view a
  best guess. Bootstrap and LOCO fractions measure internal stability across chromosomes, not calibrated
  probabilities of correct parentage; message weights, state supports and λ are not calibrated either. Resampling
  conditions on the paintings, the panel and λ, all of which use every chromosome.
- The model is a bounded composite and loopy approximation, not exact pedigree marginalisation: the genetic score
  pools two weightings of the same data, the orientation uses chromosome-level summaries (a missing-at-random
  callability correction, not common-interval recounts), and the cavity messages exclude immediate feedback but not
  every loop.
- Direction evidence depends on the paintings. Related samples of equal junction burden receive little orientation
  evidence and stay unreleased rather than guessed; design chronology is exempt.
- Eligibility is an assumption, not data: a wrong generation label or an over-restrictive constraint takes a true
  parent out of the universe, and the model can then report at most an unobserved parent. Real crosses have no
  individual-level trio truth, so real-data calls cannot be scored for accuracy; supervised validation uses
  simulations with a known pedigree, and real data are judged by internal consistency, chromosome-level stability,
  competing-candidate margins and explicit unresolved outcomes ([validation](../validation.md)).
- Known failure modes: fewer than 5 informative chromosomes (nothing released), a true parent pair outside the
  retained-parent screen (the M2 evidence is a lower bound; see `search_diagnostics.csv`), unsampled parents, closely
  related alternatives, long feedback loops and highly structured missingness.
- The exclusion release assumes calibrated, independent read errors and correct supported directions. Samples
  flagged by the contamination QC still enter it as targets and candidates; the required concordance with the
  complete view limits, but does not exclude, the effect of their miscalibrated likelihoods.
- One current record per product: changing any pedigree setting on an existing run needs a separate output root.

## Genome-wide consistent M1 floor in 2.3

Let S(c,p) be the sum of raw chromosome M1 scores for eligible parent p of child c,
and B(c) its maximum over eligible parents. The retained set includes every parent
with S(c,p) >= B(c) - 60 T(sum of informative markers), including equality.
T(m) = max(ceil(m / markers_per_information_block), 1)^information_tempering_power
uses the existing information temperature. This genome set is fixed before
calibration and resampling. Retained identities keep their raw chromosome scores.

For chromosome j, let b(c,j) be its maximum raw eligible score and T_j its temperature.
Every excluded identity uses the same chromosome profile
f(c,j) = b(c,j) - T_j / sum(T) * (sum_j b(c,j) - cutoff(c)).
Its sum is the genome cutoff. This profile feeds ranks, calibration, state/family
inference, informative tests, path/cavity evidence and resampling consistently.
Identity multiplicities, contamination and full prior denominators remain.
The transform is applied once; applying it again would change the model.

The default certified screen derives this same profile with exact fallback for
unproved candidates. Compact-screen and fixed-score records have separate keys
from legacy raw records. M2 kernels are unchanged; where exactly one parent is
available in a component, the fallback uses freshly computed raw component M1,
not the fixed genome profile. All retained-parent pairs enter the uncapped M2 panel.

Four-dataset checks preserve the adopted exact model's released fields, all 4,060
resample records and 60 CSV files; bounded default-stage and missing-parent fixtures
also verify fresh M2/exposure scores and resume. Dense downstream migration arrays
remain. These tests establish scientific identity, not sub-quadratic memory or a
whole-pipeline scaling result.
