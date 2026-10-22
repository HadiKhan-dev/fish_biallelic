# Founder identity across joins

A founder row labels a reconstructed path within an assembly component. It does not identify a
sequenced parent, and a generation label does not establish parentage. Two components' row numbers
have no implied correspondence. Within a component, the inferred correspondence can also be wrong
or unresolved even when the local allele sets are accurate.

Assembly founder-path refinement is disabled in 2.7. The refinement examples below are historical
results and research questions for 2.8; they do not describe an active 2.7 stage. Downstream founder
allele refit remains a separate stage.

## Shared paths and long-range evidence

Two chromosome paths can copy different rows in early child components, the same row through one
or more middle children, and different rows again afterwards. Exchanging their suffixes after that
shared stretch preserves every local row multiset. When the linker's maximum gap does not bridge
the shared stretch, the sum of their path-linkage scores is also unchanged: every fitted edge still
appears with the same multiplicity. This is a structural limitation of the short-range summary,
not evidence that the complete sample data cannot distinguish the correspondence.

The join scores complete sample paths using genotype emissions and a switch penalty. Samples
observed on both sides of a shared stretch can favor one correspondence even when the summed
linkage scores tie. Conversely, that preference is conditional on the switching model, the
available local rows and the search neighbourhood. A large penalized Viterbi margin is not a
calibrated probability of biological identity. The score can change with its switch penalty;
changing a penalty to agree with a reference is a model change requiring independent validation.

Unknown-allele integration, binning, sample eligibility and missing local founders can each affect
these decisions. They should be tested separately on fixed inputs. A complete-site, unbinned
counterfactual that retains a preference rules out binning and predictive unknown emissions as
necessary causes of that particular preference; it does not establish that all discarded evidence
was irrelevant.

## What a released break means

The existing release reader restricts founder-export continuity using represented, evaluated
assembly alternatives. It compares their prepared-row path multisets, retains common local content
and reports identity boundaries where the alternatives disagree about correspondence. These export
segments leave inference components and sample phase sets unchanged. A computational verification flag
or an unperformed continuation alone supplies no evidence for a new break. An absent ambiguity
record does not certify that the global correspondence has been identified.

Neither a tied short-range linker score nor a shared-row stretch alone justifies an automatic
identity break. Such a rule would also discard valid long-range sample associations. Likewise,
weak-junction scores are diagnostics under their stated objective, not calibrated identity
confidence. A broader uncertainty policy needs a declared model, controls with identifiable links
through shared sequence, ambiguous controls, and simulations with rare founder copies. Related
samples must not be treated as independent replicates when calibrating support.

## Diagnosing a suspected exchange

Use whole-component matching and independently block-rematched metrics on the same panels. Their
difference distinguishes long-range correspondence from local allele-set quality, but it does not
identify a unique causal join. Follow the source row paths through the hierarchy and recover the
actual selected incoming branch, its shared usable-site masks and its sample cohort before replay.
A nominal child representative may differ from the branch that produced a saved parent.
In 2.7, joined paths flow directly to the next assembly level and final output; downstream founder
allele refit is separate. For historical refinement-enabled runs, distinguish the raw join from
fixed-count refinement, subsequent count decisions and founder refit. Refinement could change a
correspondence that the raw join preserved, so a raw-panel swap score does not assess a different
historical refined panel.

Score the selected panel and a competing correspondence on that fixed frame, retaining per-sample
contributions and both objective and linkage margins. A better feasible panel under the same
objective demonstrates a search opportunity; it does not show that the model preferred the wrong
panel. Check whether the correction belongs to the evaluated search neighbourhood before
attributing it to a guard or a convergence rule. Historical refinement used fine/coarse proposals;
its certification was relative to that neighbourhood, not a claim of global optimality.

Use simulation truth for accuracy and missing-inclusive founder metrics. Real-data pseudo-reference
comparisons measure consistency, especially when the reference shares evidence with inference.
They do not supply known parentage or justify selecting a model, threshold or identity break to
match the reference. Any output-changing repair must also check genotype missingness/errors,
pedigree edges and phase switches under its predeclared validation gates.

Context refit currently conditions on an assembled representative. A sidecar of unresolved search
or count histories is not a posterior distribution over complete context panels. Marginalizing
those histories would require defined probabilities and established correspondence; retaining the
sidecar alone does not perform that inference.

## Shared inherited crossovers during refinement

The historical refinement model scored each observed sample with its own switching penalty. Descendants can inherit
one ancestral crossover, so many samples can support the same suffix exchange without representing
many independent recombination events. Their reads still provide separate emission evidence; the
problem is repeated switching regularization, not duplication of those reads.

In a truth-known f3 example, one accepted exchange gains about 208.55 log units while lowering
emission fit by 71.45 and saving 14 net sample-path changes. Its 24 positive contributors comprise
one observed ancestor and 23 descendants carrying the same inherited crossover. Nine opposing
samples carry continuous founder paths. The exchange creates 5,302 long-range founder errors that
remain in final assembly. This is a concrete limitation of the objective even though its preference
is strong and survives removal of any one positive contributor. Objective robustness under an
independent-sample calculation is therefore not sufficient evidence of biological correctness.

The retained refinement code's coarse-content guard protects against some absorption of shared crossovers. Extending
it directly to suffix exchanges is not a validated repair: on saved before/after panels it rejects
both the f3 error and a beneficial N80 exchange. A veto would favor the incumbent correspondence;
it would not establish that correspondence or release uncertainty. A historical guard disagreement
also need not remain unresolved after later content edits or compensating exchanges.

Real component boundaries already prevent cross-component founder identity and sample phase
continuity. An export annotation alone does not provide these inference semantics. A new automatic
break criterion must establish which correspondence is unresolved, propagate that withdrawal through
downstream stages, and report lost continuity as well as errors. Projecting one fitted component
into pieces must not present its full-span count decision as independently verified local counts.
The shared-junction diagnostic does not supply such a validated criterion. Refinement is disabled
in 2.7; this limitation remains a design question for 2.8, with no gain or contributor cutoff adopted.

## Evaluating accepted refinement moves

A move's immediate truth effect and the final trajectory's effect are different quantities.
Accepted exchanges can temporarily worsen whole-component matching while enabling later repairs.
Two exchanges can cancel up to one global row permutation; intervening content edits can still
improve the panel. Similarly, a local count deletion can lose a distinct founder path that the next
hierarchy level reconstructs. Report whether each loss persists to the final product before
proposing a veto. A frozen-panel classification does not predict a changed adaptive search.

For exchanges, report objective and emission changes, net switch savings, positive and negative
sample contributions, their concentration, and distinguishing sites on both sides. Samples with
potential observations on both sides are not necessarily informative carriers of the exchanged
pair. Effective contribution counts measure concentration, not the number of independent families
or ancestral meioses. Site counts do not count independent transmissions either.

For count changes, separate the exact deletion from the subsequent fixed-count refit and their
combined effect. Use one-to-one truth matching with explicit missing founders and report both
whole-component and block-rematched wrong-or-missing loss on the same sites. A row with no private
block cell need not duplicate another whole path: different rows may cover its cells in different
blocks. Do not sum embedded exchange effects into the deletion-plus-refit total again. Keep
hierarchy levels, terminal components, cleanup removals and count decisions separate, and retain
zero-event components in the population denominator.

Use complete selected-fit ancestries, including restored best-objective fits and the actual chosen
direction. Replay missing states through the native path and require the saved decisions, sweep
objectives and endpoints to agree before interpreting reconstructed moves. Freeze development
choices before inspecting held-out effects. Overlapping levels and adaptive moves are dependent;
descriptive counts and gain/support strata do not by themselves establish an error rate, a safe
threshold, or an improvement under an intervention.

## A local veto does not establish final identity

A diagnostic shared-event count can profile breakpoint uncertainty without deciding how much to
penalize an inherited event. For each directed founder transition, retain the contiguous interval
of breakpoint positions preserving diploid dosage on informative sites and find the minimum
number of common points covering those intervals. This is a sufficient equivalence construction,
not an enumeration of every equal-likelihood relocation. This is an origin-complexity proxy, not a reconstructed genealogy.
Coincident events need not share an ancestor. Disjoint changes of both diploid members require
separate treatment; they cannot silently be counted as one resolved transmission.

A native traceback alone is insufficient for this comparison: equally scoring paths can have
different shared-event counts. The investigated constraint requires a conservative certificate
over the native optimal-path class, including existing numerical tie tolerance. An uncertified
count is a conditional witness, not proof of nonidentifiability or a reason for an identity break.
When both panels certify, an emission loss with no decrease in shared-event count is dominated
for every nonnegative coefficient on that proxy. This does not optimize a coupled ancestral model
or calibrate a confidence level.

On the f3 example, the harmful exchange lowers emission fit by 71.45 while increasing the
certified shared-event count from 10 to 13. Vetoing it retains a truth-correct fixed-K seed. The
complete original count search nevertheless reaches exactly the original erroneous K6 panel
through another deletion seed, using one fine and one coarse content move and no exchange.
The two vetoes occur in the forward and reverse runs of the same row4 deletion fit;
that fit is absent from the selected ancestry. The final panel still has 5,304
wrong-or-missing founder alleles, of which 5,302 are correspondence excess over block matching.
The altered deletion route also changes the reported count ambiguity despite identical final
founder rows. Thus the fixed-K improvement is not a validated repair.

The final erroneous panel and the retained seed both certify, but the final panel improves
emission fit by 11.86 while increasing shared-event count from 10 to 13. Neither dominates: choosing
a penalty to reverse this result would introduce a new biological scale. The existing 20-unit
penalty is per observed sample-path change and cannot be transferred without justification to
a distinct ancestral event. Nor does the founder-count criterion automatically apply to this
same-K comparison: its cost and private-sequence shadow concern adding or removing a founder.
A useful replacement must specify its ancestry assumptions, account for related samples and
adaptive endpoint selection, and pass independent downstream validation. The investigated
constraint is not adopted. The historical refinement model's preference remains conditional on its
documented assumptions; its persistent failure is an unresolved question for the 2.8 rebuild.

## Separating rollback evidence from refinement acceptance

The following comparisons concern saved refinement-enabled runs and candidate rollback policies,
not the refinement-off 2.7 pipeline.

A computational stop can retain a component's incoming panel and its previously evaluated
alternatives. That intervention is useful for diagnosis, but the work limit is not evidence that
the retained biological correspondence is correct. Compare the complete incoming and returned
panels; a partial fit or window recorded at the stop is not necessarily either endpoint.
Distinguish a direct comparison on identical prepared evidence, incoming rows, context and
alternatives from downstream differences propagated from an earlier rollback. Equal summary
scores do not establish equal inputs.

Keep original component spans fixed when comparing continuity. A candidate split makes the
original whole-span correspondence unavailable; independently rematching its pieces cannot
establish that correspondence. Preserve those lost opportunities alongside conditional metrics
on the spans that remain comparable. Assembly panels, later founder refit and released export
have different content and masking rules, so improvements at one endpoint do not imply
improvements at every endpoint.

Missing-inclusive one-to-one matching has two distinct sources of missingness: unknown alleles
on matched inferred rows, and dummy rows for reference haplotypes without an inferred match.
Report both, along with raw unknowns across all inferred rows, unmatched inferred rows and their
called-allele accuracy. These have different denominators. Extra inferred rows may improve the
one-to-one target score without being penalized by that score; nearest-truth accuracy over every
inferred row and represented-versus-absent founder truth at each site provide necessary complementary views.
A sequenced real-data reference row need not represent a founder transmitted to the sampled cross.

In a saved TroMau comparison, 165,000 of 165,676 fewer missing reference opportunities came from
eliminating dummy-reference penalties on two chromosomes. Raw unknown alleles on the shared QC
site axis instead increased by 34 across a larger inferred panel. That headline does not show
that refinement had uncalled 165,000 alleles. Other components do show increased called
pseudo-reference disagreement after deletion and refit, so the complete effect cannot be
explained away as dummy matching either. TroMau remains a dependent consistency check; simulated
truth and later-stage outcomes decide whether a scientific intervention is supported.

A robustness criterion must compare the quantity that actually selects the model. Fixed-K
penalized path score, founder-count support and a shared-ancestry uncertainty model are different
quantities. A deletion can lower the best path score while satisfying the current count criterion.
Requiring every count change to have a positive path-score gain would silently change that model.
Likewise, resampling related observed individuals independently does not calibrate uncertainty in
an ancestral correspondence. Computational difficulty, small gain, few contributors and an
improved real-data matching score are diagnostics, not acceptance rules on their own.

## Family stress test of a shared-event Pareto veto

A bounded test of 96 panels used 48 shared-recombinant cases and 48 cases with independently
placed breakpoints. Each contained 48 samples and 320 sites, with K = 6 or 12, two depths,
three founder-missingness fractions and four seeds. In a shared case, 40 descendants were
conditioned on the same inherited recombinant gamete. The fixtures did not generate complete
pedigrees; the independent cases also remained enriched for the same founder pair. These are
objective stress tests, not a calibrated sample of pedigrees.

The proposed veto rejected a move if its allele-emission evidence fell and its minimum
resolved-transition interval-cover count rose. Both quantities came from the native optimal
sample paths. The primary comparison used the all-kept-site, seven-class predictive evidence
with the existing 20-nat unordered-state penalty; a complete-site comparison used the same
inputs. All seven candidate scores, switch counts and founder metrics reproduced the saved
reference in every case. The event counter used predictive-class equality and preserved
unknown-allele source identities. Truth and family labels were used only for reporting.

The veto retained the original panel in 0 of 48 shared cases under either objective. It changed
no selected panel in the independent cases either. Every all-site shared-case winner was a
suffix exchange at the generating breakpoint: 37 increased conditional emission evidence;
the remaining 11 decreased both evidence and the event-count proxy. The earlier variant that
also vetoed evidence loss at an unchanged count likewise rejected no improving proposal.
The diagnostic allowed representative-path vetoes without an all-optimal-path certificate,
so it gave the rule more latitude than a certified implementation would have.

Noisy independent paths can fragment one inherited event into several non-overlapping exact
emission-preserving intervals. In one fully called K = 6 case, the original 40 sample changes
became six proxy events. The absorbed panel had five sample changes and five proxy events;
its native score gained 696.85 nats despite losing 3.15 emission nats, and founder errors rose
from zero to 140. Exact-equality intervals are not uncertainty intervals for a shared origin.
Also, these candidates preserve the per-site founder-allele multiset: their conditional
emission differences reflect paths selected by the switching penalty, rather than an
independent comparison of ancestral phase.

Guarding fine/coarse edits, births, deletions and refits with this same predicate cannot block
a directly allowed winning suffix exchange. The rule failed its efficacy prerequisite;
no production veto or new biological identity break was adopted. A future rule must also
survive complete adaptive count search, including seed response, cleanup, direction choice,
restored fits and terminal alternatives. Fixed-count rescue alone is insufficient.

## Requirements for pedigree-aware switch accounting

A successor model should attach recombination to the transmission edge where it arose.
Let a binary selector choose a parent's transmitted homologue along that meiosis, while
founder ancestry follows the selected parental homologue. An inherited parental ancestry
change then propagates through descendants without introducing a new selector change in each
child. Reads still contribute separately once per sample/site. Distinct meioses remain
separate origins even when their crossover coordinates coincide.

Joint inference must estimate common breakpoint positions and uncertain relationships from
the observations, rather than intersect independent exact-equality intervals or assume cohort
labels identify families. Missing and unsequenced parents remain possible. Initial comparison
with the all-site robust evidence should hold that evidence term fixed and explicitly call
the result a generalized score; a calibrated likelihood requires a normalized observation
model. Joint integration of shared unknown founder alleles is a separate model decision.

Structured factor-graph inference with K-state ancestry variables and binary transmission
selectors is a possible approximation. Local observation work can be quadratic in K and
copying factors can exploit equality structure. General pedigree elimination can be more
expensive: a production implementation must expose its approximation, convergence residual
and cap, validate against tiny exact pedigrees, and establish at most cubic K dependence for
the complete proposal/count sweep. A cheap per-panel calculation alone does not prove that
bound when there are quadratically many candidate panels.

The coupled model must define consistent founder-count comparisons; current independent
per-sample private-sequence margins cannot be transplanted without justification. Test both
truth-correct and erroneous incumbents, beneficial rare-founder births, missing parents,
independent coincident events and every adaptive search route. Known-pedigree oracle success
is diagnostic only; the actual uncertain-parent estimator needs held-out truth validation.
Where evaluated terminal explanations disagree on correspondence, release ambiguity with its
original-span continuity cost. Computational limits and unperformed search remain separate
from biological nonidentifiability. This is a design requirement, not an implemented repair.
