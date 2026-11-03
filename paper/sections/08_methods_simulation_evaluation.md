## Simulation with recorded truth

Accuracy is measured only on simulated crosses, because the real cross has no individual-level trio truth. A simulation (`[input] kind = "simulation"`) generates allele depths from founder templates and a known pedigree; `haplotypes evaluate` compares the finished run with the recorded truth. The generating read model, crossover process and map are configured independently of the inference model and map. Inference receives only sample names and REF/ALT counts, as from a VCF; the truth, including cohort and pedigree descriptions, is read only by evaluation.

### Founder templates

A contig's template gives marker positions and, for each of $H$ founder haplotypes ($H$ even and equal across contigs), a probability for each allele. The `haplotypes templates` command builds templates from an observed VCF. The reported validation uses pre-existing empirical templates and a constructed high-founder-count set, whose recorded provenance and contigs are given in Supplementary Information, section S6. It runs the pipeline's block discovery on fixed-error evidence (filters and read calibration disabled) in 100 kb windows at a 50 kb stride, treating uncalled alleles as neutral (0.5, 0.5). A simple linker chains windows into up to six contig-length paths by penalised dynamic programming over junction mismatch scores, merging overlaps by adding signed allele information. Templates thus have realistic marker density and haplotype structure but are approximations.

Founder alleles are the more probable template allele at each marker. Exact ties, including neutral cells, are resolved by a per-contig random stream, so no reference bias enters the truth. Consecutive haplotypes are paired into $H/2$ diploid founders.

### Pedigrees and partial sequencing

The default design has successive non-overlapping cohorts (default sizes 20, 100 and 200): cohort 1 descends from the founders and each later cohort from the one before. Each child receives an ordered pair of distinct, sexless parents drawn uniformly from the previous generation, so sibships arise by chance. From a configured cohort onward (default 3), each child is, with probability `backcross_fraction` (default 0), instead a backcross of a uniformly chosen previous-cohort member with one of its own parents. All contigs share one pedigree. Alternatively, an explicit pedigree table gives each individual's two earlier parents and an observed flag; its founders take the template founders in order and may be sequenced, and its sex and generation columns are truth descriptions read by no inference step.

Partial sequencing follows pedigree construction: optionally only listed cohorts are eligible, and a uniformly random $\lfloor\theta n_{\mathrm{e}}\rfloor$ of the $n_{\mathrm{e}}$ eligible individuals is kept in pedigree order ($\theta$ = `observed_fraction`, default 1; at least three must remain). In the cohort design founders are never sequenced, so cohort-1 individuals, like any child of an unsequenced parent, have missing parents.

### Meiosis

Each gamete starts on a uniformly chosen parental homologue and switches at each crossover. Without a generating map, crossovers form a Poisson process along the physical coordinate at the fallback rate $\gamma$ (default 5 cM/Mb, i.e. $5\times10^{-8}$ Morgans per bp), drawn as exponential gaps rounded up to whole base pairs. With a generating map, the number of crossovers between the first and last marker is Poisson with mean equal to their map distance $x$ in Morgans, and their positions are uniform in genetic coordinates and mapped back to base pairs; contigs absent from the map use $\gamma$. Both are no-interference processes, consistent with the Haldane map function [@haldane1919]. Mutations arise at $\mu=10^{-10}$ per bp per meiosis and flip the allele of the next marker without changing ancestry.

### Reads

Each sequenced sample $i$ has depth factor $u_i=1$ unless the between-sample coefficient of variation $v$ is positive; then $u_i$ is lognormal with mean one and coefficient of variation $v$, renormalised to average exactly one. Optional dropout zeroes one contiguous tract of $w=\lfloor\phi L\rfloor$ markers per sample at a uniform start ($\phi$ = `dropout_fraction`), with depth scaled by $1/(1-w/L)$ to keep the mean depth $\lambda$ (default 5). In the default independent-site model,

$$d_{ij}\sim\mathrm{Poisson}\left(\frac{\lambda u_i}{1-w/L}\right),\qquad a_{ij}\mid d_{ij}\sim\mathrm{Binomial}\bigl(d_{ij},\,p_{g_{ij}}\bigr),$$

where $d_{ij}$ and $a_{ij}$ are the total and alternate read counts of sample $i$ at site $j$, $g_{ij}$ the true dosage, $p_0=\varepsilon$, $p_1=b$ and $p_2=1-\varepsilon$, with error rate $\varepsilon$ (default 0.02) and heterozygote alternate-read probability $b$ (default 0.5; other values simulate allelic imbalance). A positive read length $\ell$ selects a paired-end fragment model instead. Fragment starts form a per-sample Poisson process scaled by the expected bases covered per fragment, so each marker keeps the same Poisson depth marginal while nearby markers share reads. Insert lengths are normal (default mean 375 bp, sd 150 bp), rounded and clipped to [50 bp, mean + 6 sd]; each fragment copies one uniformly chosen homologue and is read over $\ell$ bp from each end, overlapping mates counted once. Allelic imbalance converts a read of the disfavoured allele at a heterozygous marker with probability $|2b-1|$, and errors flip reads at homozygous markers with probability $\varepsilon$. In both models heterozygous reads carry no sequencing error and alternate-read probability $b$.

### Recorded truth and random streams

Per contig, the simulator records the true founder haplotypes; the maternal and paternal alleles of every sequenced sample; ancestry chunks, maximal half-open intervals of constant maternal and paternal founder origin; and each sequenced child's gametes with parent, slot, cohort and crossover positions. Chunk bounds are stored as integers, truncating fractional crossover positions, so truth can disagree at a boundary marker. A manifest records the requested and realised seed, contig order, sequenced samples and the generated pedigree.

All randomness comes from named streams with fixed generator construction and draw order: per-contig founder streams; a master stream that seeds a Mersenne Twister structural generator [@matsumoto1998] and, per cohort or pedigree depth and before any scheduling, draws one seed per contig and child; a separate observation stream; and one read seed per contig, drawn for every contig even on resume. NumPy streams use PCG64 [@harris2020; @oneill2014]. A configuration and seed therefore give the same data at any worker count.

## Evaluation against recorded truth

`haplotypes evaluate` scores contigs independently and pools counts and spans, recomputing ratios and N50 values from pooled totals; undefined ratios are null. It scores the founder product of any stage (discovery blocks, three local-selection rounds, assembly levels 1–4, refinement, painting, founder refit) on its original component and row axes, plus pedigree, family phase and recombination. Per-sample local-ancestry paintings and relatedness families are not scored against truth. The metrics are not interchangeable.

### Founder panels

A founder-panel component $c$ has $K_c$ rows over $L_c$ markers (calls 0, 1 or missing). For row $k$ and true founder haplotype $f$, $e_{kf}$ counts the called markers where they differ and $m_k$ the row's missing calls. Nearest-founder errors, $\sum_k\min_f e_{kf}$, are reported per million called alleles. Duplicate rows can score well by this measure without recovering every founder, so a one-to-one assignment $\pi$ minimising $\sum_k(e_{k\pi(k)}+m_k)$ is also solved by rectangular linear assignment [@virtanen2020], with rows in content order so that exact ties never depend on row labels. With $\mathcal M_c$ the matched rows, the truth-to-panel count

$$T_c=\sum_{k\in\mathcal M_c}\bigl(e_{k\pi(k)}+m_k\bigr)+\bigl(H-|\mathcal M_c|\bigr)L_c$$

charges every allele of an unmatched founder; markers outside every component add $H$ missing alleles each. A segment-wise score labels each row by its minimum-cost path through the true founders, a mismatch costing 1 and a label switch $\kappa=20$. Its mismatches and switches are reported, and nearest-founder errors minus segment mismatches give chimeric charges. A split-charged cost adds $\kappa$ per row at each component start after the first, so cutting a component cannot lower it, unlike the component-wise counts.

### Ancestry strata and identity continuity

From the ancestry chunks, $c_{fj}$ counts the sequenced homologues carrying true founder haplotype $f$ at site $j$. Matched alleles are reported by copy stratum, absent ($c_{fj}=0$), single ($c_{fj}=1$) or multiple ($c_{fj}\ge2$); these strata describe representation in the sample, not identifiability. Identity continuity labels each row with a founder in fixed 100-marker truth windows where it clearly matches one (Supplementary Information, section S6). Consecutive labelled windows of a row assert continuity, correct when the labels agree, classified as adjacent or gap-spanning and by stratum. Coverage counts pairs of nearby windows where a founder is represented and distinguishable that a single row labels with that founder in both.

### Released phase

Phase is scored per sample within released phase sets, maximal runs of the sample's sites inside one painting component not separated by its released phase breaks. Scored sites are correctly called true heterozygotes released phased. A scored site's orientation is flipped when its first released allele differs from the true maternal allele. Per set, site errors count the scored sites on the minority orientation; switch errors count orientation changes between consecutive scored sites; long switches count changes between consecutive orientation runs of at least 20 scored sites, ignoring shorter runs. Unscored sites do not split sets. Genotype errors (an incorrect allele sum) are counted separately, never as phase errors. The phased share is scored sites over correctly called true heterozygotes. N50 is taken over first-to-last scored-site spans; a truth-free count and N50 of released sets are reported alongside. Results are also grouped by each sample's number of released inferred parent edges (0, 1 or 2), not true parentage.

### Pedigree

The exact-configuration Tier-B call product is compared with the true parents restricted to sequenced samples. The supported-edge product used for phase and recombination also includes marginal edges from incompletely resolved configurations; it is scored separately in Supplementary Table S5. Edges are matched by sample name, ignoring slot order; an unsequenced true parent is not an observable edge. A configuration is exact when the called parent state (zero, one or two observed parents) and the parent set both equal the truth; an unresolved call is never exact. Correct, extra and missing parent–child edges are counted separately.

### Recombination

Map calls are scored only on inferred parent–child edges that are true gametes, reporting true crossovers within the inferred informative spans, the stage's own expected crossover count and exposure (bp × meioses), the called intervals, and the fraction containing a true crossover under an open-left, closed-right rule, for called and for reported intervals, with median widths. Hit rates are read with widths, because wider intervals hit more often.
