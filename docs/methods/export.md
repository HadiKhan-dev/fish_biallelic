# Released outputs and reports

Outputs format checked typed views. Founder carrier withholding and assembly release decisions occur upstream; export performs no search. N is samples, L written sites, K founder columns and E parent edges. Column order, missing values, tuple encodings, phase-set semantics and index choice are public contracts. See the output dictionary for the exact file formats.

The sections below identify semantic units, not separately configurable pipeline steps. [Outputs](../user/outputs.md), [record contracts](../reference/records.md) and [costs](../developer/costs.md) describe their public boundaries.

<a id="ou01"></a>
## ou01 Export site table: rows written, FILTER bits, REF/ALT, loader-drop accounting

Build ReleasedSites through the ObservationSource contract, binding source alleles, positions and contig metadata. Require every evidence position to occur exactly once on the admitted source axis and require applied filter bits to agree with kept-site membership. Emit full-length FILTER names in rule order; kept-only rows use the specified missing FILTER encoding. Duplicate-position loader drops remain separate from scientific site filters. Export formatting reads this product through results.

Implementation: [contracts/release.py](../../haplotype_reconstruction/contracts/release.py), [outputs/export/records.py](../../haplotype_reconstruction/outputs/export/records.py), [stages/sites/tasks.py](../../haplotype_reconstruction/stages/sites/tasks.py), [stages/inputs/vcf_alleles.py](../../haplotype_reconstruction/stages/inputs/vcf_alleles.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou02"></a>
## ou02 Common allele at r5-only sites (founder-VCF fill rule)

At sites removed only by r5, choose the common allele from summed nonnegative AD counts, using GT only when AD is absent or wholly missing in its sentinel form. Use the exact integer majority rule. Fill every eligible founder column with the same common allele, but suppress fill for release-withheld rows. A site failing another rule gets no fill; this fallback carries no founder-specific evidence.

Implementation: [stages/sites/tasks.py](../../haplotype_reconstruction/stages/sites/tasks.py), [stages/inputs/vcf_alleles.py](../../haplotype_reconstruction/stages/inputs/vcf_alleles.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(#r5-only sites x N_counted); O(M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou03"></a>
## ou03 Unobserved founder-allele withholding (painted-carrier rule)

Before founder segmentation, withhold a called allele unless an eligible painted carrier supplies observed support through a named or pooled candidate row. Check both tracks, ignore out-of-range members and preserve sample/site alignment. This scientific release rule is computed upstream of formatting. It uses refit calls when available, so a painting support report need not predict every final withheld allele.

Implementation: [science/assembly/release.py](../../haplotype_reconstruction/science/assembly/release.py), [stages/release/compute.py](../../haplotype_reconstruction/stages/release/compute.py), [outputs/export/run.py](../../haplotype_reconstruction/outputs/export/run.py), [interface/cli.py](../../haplotype_reconstruction/interface/cli.py), [science/painting/release.py](../../haplotype_reconstruction/science/painting/release.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(L · N · 2 · |member set|). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou04"></a>
## ou04 Founder VCF: release segments as columns, spans, cells, header

Render each release segment as its declared founder columns over its own inclusive site span. Insert kept panel calls and permitted r5-only common alleles, retaining dots outside a column’s span. Require exact site alignment and kept membership. Emit metadata and column names in contract order, then indexed VCF/BCF records. Chunking changes memory use only; overlapping spans still fill independent columns.

Implementation: [outputs/export/run.py](../../haplotype_reconstruction/outputs/export/run.py), [outputs/export/founders.py](../../haplotype_reconstruction/outputs/export/founders.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(M · (C + K_active)). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou05"></a>
## ou05 Sample phase VCF: GT:PS encoding, phase sets, unphased and filtered sites

Restore input sample order and encode released genotype alleles with GT:PS. A phased set begins at its first kept site, not necessarily a heterozygote; unphased sites use a missing PS without terminating the set. Component/set boundaries determine later PS values. Filtered rows are wholly missing. An unphased genotype is written in ascending allele order, missing first (`0/1`, `./1`), because its order conveys no phase assertion. Phased cells keep their released allele order.


Implementation: [outputs/export/run.py](../../haplotype_reconstruction/outputs/export/run.py), [outputs/export/samples.py](../../haplotype_reconstruction/outputs/export/samples.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · M). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou06"></a>
## ou06 Founder identity flags (duplicate-partner changes)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Render duplicate-partner identity changes from the released typed flags. Associate a flag with the segment holding its right block and translate rows through that segment’s source-row map. At a release boundary, name left partners using the left segment. Report the left block’s last marker and right block’s first marker; skip flags whose right block has no release segment. Flags qualify identity continuity, not allele correctness.

Implementation: [science/assembly/release.py](../../haplotype_reconstruction/science/assembly/release.py), [stages/release/compute.py](../../haplotype_reconstruction/stages/release/compute.py), [outputs/export/founders.py](../../haplotype_reconstruction/outputs/export/founders.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(B K²); O(K²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou07"></a>
## ou07 Founder track TSVs (tracks product)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Write one founder-track table with chromosome, position and local haplotype columns, preserving segment/site order and the specified dot/allele encoding. Compression and site chunking do not change cell text. Tracks are selected export products, not another inference pass.

Implementation: [outputs/export/founders.py](../../haplotype_reconstruction/outputs/export/founders.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(K L). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou08"></a>
## ou08 Export orchestration: manifest, record identity, resume, index and atomic placement

Declare one formatting task per selected contig and reuse an export only when its identity and every listed file match. Place data/index files before publishing completion, remove superseded files only after successful rendering and order manifest entries by configured contig. Choose CSI versus TBI from the last written position. Standalone export requires an empty destination and consumes checked released records; it has no source --vcf option.

Implementation: [outputs/export/records.py](../../haplotype_reconstruction/outputs/export/records.py), [outputs/export/run.py](../../haplotype_reconstruction/outputs/export/run.py), [outputs/export/task.py](../../haplotype_reconstruction/outputs/export/task.py), [interface/cli.py](../../haplotype_reconstruction/interface/cli.py).

Parameters by name: `TASK_ALLOWANCE_BYTES`, `enabled`. Values and provenance are in the [generated parameter reference](../reference/parameters.md).

Cost and scope: O(N M + M · columns). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou09"></a>
## ou09 Sample-axis order of published outputs (canonical order in records, input order outside)

Keep scientific records on the canonical sample axis and restore the recorded input axis at external sample-facing boundaries. Stable-sort tables by Sample when present, preserving unknown-name order after known names. Assign sample-switch indices from the input position. Validate the permutation instead of silently reordering one array without its companions.

Implementation: [contracts/samples.py](../../haplotype_reconstruction/contracts/samples.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou11"></a>
## ou11 Report writer conventions and publication schedule

Render reports on every invocation from the currently available checked records. Use typed post-guard pedigree views for released tables while keeping the complete/diagnostic views’ documented pre-guard meanings. Read each contig’s phase and recombination products through results and publish tie summaries after completion. Output formatting never reaches into assembly internals or changes science.

Implementation: [outputs/reports/tasks.py](../../haplotype_reconstruction/outputs/reports/tasks.py), [outputs/reports/tables.py](../../haplotype_reconstruction/outputs/reports/tables.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · chunks · K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou12"></a>
## ou12 Painting released view and carrier-support tables

Enumerate painting components, founder rows and fixed site windows in order. Summarise carrier support only over called painted alleles and retain missing minima/means when no allele is called. Use named TrackStatus meanings and preserve pooled-candidate counts. These tables describe the painting release; final founder export may additionally reflect refit and assembly withholding.

Implementation: [outputs/reports/painting.py](../../haplotype_reconstruction/outputs/reports/painting.py), [science/painting/release.py](../../haplotype_reconstruction/science/painting/release.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · chunks · K). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou13"></a>
## ou13 Painting and assembly diagnostic tables

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Format painting, count and join-search diagnostics from typed report views. Sum overlapping join-search records once per component and keep context-assembly feedback separate. Compute sample switch ratios relative to the eligible cohort median, without generation stratification. Missing constituent counts stay missing where the column contract requires them. Reports describe retained search decisions and do not rerun them.

Implementation: [outputs/reports/painting.py](../../haplotype_reconstruction/outputs/reports/painting.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N · components). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou14"></a>
## ou14 Relatedness tables and summary

Write relatedness family/support products in their declared row order, rendering unsequenced parent slots as empty names. Remove a `relatedness/pairs.csv` found in the output directory: the stage publishes no pairwise table ([relatedness](relatedness.md#pairwise-identity-by-descent)).

Implementation: [outputs/reports/relatedness.py](../../haplotype_reconstruction/outputs/reports/relatedness.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N²). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou15"></a>
## ou15 Pedigree tables and summary JSON (pass-through)

Format all pedigree tables from PedigreeViews, preserving ordered columns, tuple representations, missing encodings and ParentSlotBasis. Stable-sort Sample tables to input order; leave other table axes unchanged. Write summary fields in their specified order. These report JSONs permit the documented nonfinite values, unlike evaluation’s strict JSON.

Implementation: [outputs/reports/pedigree.py](../../haplotype_reconstruction/outputs/reports/pedigree.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Linear in emitted cells; most relationship tables have N rows, exclusion tables have one row per scored dyad.

<a id="ou16"></a>
## ou16 Quality tables (sample QC exclusion, inherited segments)

Write sample-quality and inherited-segment tables from the quality products. Restore input order for samples and keep segment record order. When the sample table is absent, return before writing either table. Preserve separate fit, prior and final exclusion meanings rather than interpreting every flagged sample as known contamination.

Implementation: [outputs/reports/evidence.py](../../haplotype_reconstruction/outputs/reports/evidence.py), [outputs/reports/quality.py](../../haplotype_reconstruction/outputs/reports/quality.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(N). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou17"></a>
## ou17 Stage summary tables (founder_refit, family_phase)

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Render founder-refit and family-phase summaries from their typed report records, preserving contig order, named counters and missing values. Phase history-step counts retain their documented repeated-start convention. Timings are measured runtime diagnostics supplied separately; unavailable durations are not invented.

Implementation: [outputs/reports/tables.py](../../haplotype_reconstruction/outputs/reports/tables.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(C). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou18"></a>
## ou18 Recombination tables and plot

Render crossovers and meioses in edge/call order, with names from the product axis, and concatenate contig summaries/junctions in configuration order. Preserve inclusive reported intervals and missing rates without exposure. Plot expected and called rates with effective meioses on physical coordinates. The figure and tables are views of the same conditional map, not independent estimates.

Implementation: [outputs/reports/recombination.py](../../haplotype_reconstruction/outputs/reports/recombination.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: O(E · B + calls). These terms describe the unit’s work/storage dimensions; iteration and candidate limits are the named parameters above.

<a id="ou19"></a>
## ou19 Ties and ambiguity report

**Accretion:** retained with its declared behaviour; see the [accretion register](../developer/accretions.md).

Collect typed tie counts by configured contig, stage/group and kind order. Summarise discovery and assembly ambiguity with the declared counter meanings and score-margin statistics. Report the explicit run convention from settings; no environment lookup selects it. Diagnostic concurrency cannot alter row order or a scientific decision.

Implementation: [outputs/reports/ties.py](../../haplotype_reconstruction/outputs/reports/ties.py).

Parameters: supplied typed inputs and the stage’s [parameter record](../reference/parameters.md); any convention is an explicit [tie policy](../developer/ties.md). This unit has no additional independent setting.

Cost and scope: Selective record-item reads plus sorting emitted counter rows; no scientific recomputation.

<!-- named mechanism anchors -->

## Named mechanisms

<a id="sites"></a>

**Sites:** see [OU03](#ou03).
