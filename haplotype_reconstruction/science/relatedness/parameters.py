"""Name the family model and search parameters.

Likelihood thresholds are in nats, support intervals in Morgans, and probability controls are dimensionless.
Method: docs/methods/relatedness.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AlleleFrequencyParameters:
    """Sitewise Hardy-Weinberg EM schedule and probability bounds."""

    iterations: int = field(
        default=6,
        metadata={
            "provenance": (
                "Hardy-Weinberg EM updates per site (iterations); fixes the allele-frequency "
                "estimation schedule."
            ),
        },
    )
    initial: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Initial ALT frequency (probability); starts the sitewise EM at equal REF and ALT "
                "mass."
            ),
        },
    )
    clip: float = field(
        default=1e-4,
        metadata={
            "provenance": (
                "Lower ALT-frequency bound and distance from one (probability); keeps genotype "
                "priors away from zero."
            ),
        },
    )
    total_floor: float = field(
        default=1e-300,
        metadata={
            "provenance": (
                "Minimum genotype-weight sum (probability); prevents division by zero in the EM "
                "posterior."
            ),
        },
    )


@dataclass(frozen=True)
class FamilyParameters:
    """Family evidence gates, objective tolerances, support units and workspace bounds."""

    minor_allele_frequency: float = field(
        default=0.05,
        metadata={
            "provenance": (
                "Minimum minor allele frequency (fraction); selects informative sites for family "
                "fitting."
            ),
        },
    )
    minimum_observed: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Minimum observed-sample fraction (fraction); excludes poorly observed sites from "
                "family fitting."
            ),
        },
    )
    family_site_step: int = field(
        default=20,
        metadata={
            "provenance": (
                "Stride through eligible family sites (sites); thins the family likelihood "
                "evidence."
            ),
        },
    )
    max_sweeps: int = field(
        default=30,
        metadata={
            "provenance": (
                "Maximum sweeps of each family search (sweeps); bounds search iterations."
            ),
        },
    )
    support_unit_morgans: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "Length of the support units (Morgans of the configured working map, DESIGN_417 D5): the family "
                "sites' axis pooled over contigs in configured order is cut into consecutive units of this "
                "length, each weighted by its Morgans in the member and parent support shares."
            ),
        },
    )
    support_tail_morgans: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "Final-piece length (Morgans) below which the last piece of the pooled support axis joins the "
                "unit before it (DESIGN_417 D5, the pedigree material rule)."
            ),
        },
    )
    rate_estimate: str = field(
        default="veto",
        metadata={
            "provenance": (
                "[genetic_map] rate_estimate (owner, 14 Nov 2026): with adopt, the estimated genome-wide "
                "recombination rate becomes the working fallback rate (contigs without a map file and stretches "
                "beyond a file's knots) when its gate passes, the supplied rate lies outside its 95 % interval and "
                "at least minimum_adoption_families informative families were used; with veto or ignore it is a "
                "diagnostic and nothing downstream reads it."
            ),
        },
    )

    @property
    def infers_rate(self) -> bool:
        """Whether this estimate may become the working fallback rate (rate_estimate = adopt)."""
        return self.rate_estimate == "adopt"
    source_populations: int = field(
        default=1,
        metadata={
            "provenance": (
                "rl21: source populations the cohort descends from (count; the [input] source_populations "
                "setting). 1: one cohort allele frequency, a latent parent is a Hardy-Weinberg draw (no ancestry "
                "model runs). 2: a cross between two diverged populations or species; a latent parent's law is the "
                "marginal over ancestry classes at source-specific frequencies. Validated in the 4.19 "
                "study at depth 5 with pure or first-generation-hybrid parents (held-out interspecies F2 "
                "simulations: 1,274 misplaced sample pairs with one source, 45 with two, 29 in the one-population "
                "control); not shown to help at depth 1 (1,114 against 1,277) or for unsequenced later-generation "
                "parents; misdeclared for one population it splits sibships. An automatic choice was tested and "
                "rejected in the 4.19 study: a cross from few founders is itself an admixture of founder "
                "haplotypes, and no admixture statistic separated the two cases with a margin."
            ),
        },
    )
    source_shares: tuple[float, ...] = field(
        default=(0.0, 0.5, 1.0),
        metadata={
            "provenance": (
                "rl21: the homologue types, each a share of source B constant over the genome (probability): a "
                "pure-A homologue, one transmitted by a first-generation hybrid (each site from A or B with "
                "probability 1/2 by Mendel) and a pure-B homologue. The classes are their unordered pairs: pure, "
                "first-generation hybrid, backcross, and F2 or later. A finer grid represents later backcrosses "
                "at (T (T + 1) / 2)^2 class pairs per family term for T types."
            ),
        },
    )
    source_clip: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "rl21: bound of the source allele frequencies away from 0 and 1 (probability). It caps what one "
                "site that contradicts a class can cost (ln(1 / clip) nats for one allele), the allowance for a "
                "frequency estimated from the few haplotypes a source contributed to the cross."
            ),
        },
    )
    source_switch_rate: float = field(
        default=4.0,
        metadata={
            "provenance": (
                "rl21: rate per Morgan at which a sample's local ancestry state is redrawn from its genome-wide "
                "shares along the working map (1/Morgan). An F2's state changes when either parental gamete "
                "crossed over: 2 per Morgan, which a redraw rate of 4 gives in the heterozygous-ancestry state; "
                "pure and first-generation samples have one state and never switch."
            ),
        },
    )
    source_state_error: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "rl21: probability given to a child's reads at a site where its local ancestry state cannot be "
                "produced by a pair of parental configurations (probability): the allowance for a wrong local "
                "state. One such site costs ln(1 / error) nats."
            ),
        },
    )
    source_fit_sites: int = field(
        default=4000,
        metadata={
            "provenance": (
                "rl21: at most this many family sites (every s-th in genome order) feed the fit of the samples' "
                "ancestry shares (sites); bounds it at O(N x 4000) per iteration."
            ),
        },
    )
    source_starts: int = field(
        default=3,
        metadata={"provenance": "rl21: seeded starts of the ancestry fit (count); the best log likelihood is kept."},
    )
    source_iterations: int = field(
        default=300,
        metadata={"provenance": "rl21: maximum EM iterations of one start of the ancestry fit (count)."},
    )
    source_tolerance: float = field(
        default=1e-7,
        metadata={"provenance": "rl21: relative log-likelihood gain below which a start of the ancestry fit stops (ratio)."},
    )
    source_seed: int = field(
        default=4190,
        metadata={"provenance": "rl21: seed of the ancestry fit's starts (integer); the fit uses no sample name or order."},
    )
    source_site_iterations: int = field(
        default=30,
        metadata={
            "provenance": "rl21: EM iterations of every family site's source frequencies at the fitted sample shares (count)."
        },
    )
    source_chunk: int = field(
        default=5000,
        metadata={"provenance": "rl21: family sites per block of the source-frequency pass (sites; memory only)."},
    )
    source_strong_differentiation: float = field(
        default=0.9,
        metadata={
            "provenance": (
                "rl21: source allele-frequency difference (probability) above which a family site counts as "
                "strongly differentiated for the misdeclaration warning. Near-fixed differences are what a cross "
                "between two diverged sources has and a one-population cohort lacks."
            ),
        },
    )
    source_minimum_strong: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "rl21: share of family sites that must be strongly differentiated, or the stage warns that the "
                "cohort does not look like a two-source cross or that its sources are not identified (share). In "
                "the 4.19 study 16 one-population cohorts declared as two sources had 0.000-0.001; the two-source "
                "cohorts whose ancestry was read correctly had 0.04-0.50; the one whose sources were not "
                "identified (weak divergence at depth 1) had 0.000. 0.01 is ten times the largest of the first "
                "group and a quarter of the smallest of the second. A warning only: no result depends on it."
            ),
        },
    )
    source_validated_depth: float = field(
        default=3.0,
        metadata={
            "provenance": (
                "rl21: estimated reads per family site and sample below which the stage warns that two sources "
                "are declared outside the validated depth (reads; the cohort median of -ln(1 - a sample's "
                "observed share of family sites)). In the 4.19 study the held-out interspecies F2 simulations "
                "met the criteria at depth 5 and failed them at depth 1 (cousin sibships merged around an aunt or "
                "uncle in a parent slot); nothing was measured between, so the bound is their midpoint. A warning "
                "only: no result depends on it."
            ),
        },
    )
    source_differentiation: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "rl21: source allele-frequency difference (probability) above which a family site is counted as "
                "differentiated in the report. A diagnostic only: no site is removed and no law depends on it."
            ),
        },
    )
    gain_tolerance: float = field(
        default=1e-6,
        metadata={
            "provenance": (
                "Family objective and support tie tolerance (nats); prevents numerical noise from "
                "choosing a placement or counting support."
            ),
        },
    )
    screen_margin: float = field(
        default=0.1,
        metadata={
            "provenance": (
                "Fixed screening-error allowance (nats); triggers exact evaluation near a "
                "screened gain boundary."
            ),
        },
    )
    screen_margin_per_site: float = field(
        default=1e-6,
        metadata={
            "provenance": (
                "Additional screening allowance per family site (nats per site); scales the error "
                "allowance with likelihood length."
            ),
        },
    )
    screen_rounding: float = field(
        default=2.0**-53,
        metadata={
            "provenance": (
                "Relative rounding factor in the screening-error bound (dimensionless); "
                "multiplied by twice the padded site count and gain magnitude."
            ),
        },
    )
    screen_padding: int = field(
        default=16,
        metadata={
            "provenance": (
                "Extra site count in the screening-rounding bound (sites); provides a fixed "
                "rounding allowance for short inputs."
            ),
        },
    )
    screen_tile: int = field(
        default=16,
        metadata={
            "provenance": (
                "Family-pair screen tile width (families); bounds workspace without selecting "
                "candidates."
            ),
        },
    )
    sum_bytes: int = field(
        default=32 * 1024**2,
        metadata={
            "provenance": (
                "Target member-sum temporary budget (bytes); bounds the site chunk for "
                "accumulating family terms."
            ),
        },
    )
    pair_batch: int = field(
        default=262144,
        metadata={
            "provenance": (
                "Packed pair gains evaluated per batch (pairs); bounds exact-evaluation "
                "workspace."
            ),
        },
    )
    prefetch_batch: int = field(
        default=64,
        metadata={
            "provenance": (
                "Family fits prefetched per batch (families); bounds cached evaluation work."
            ),
        },
    )
    term_floor: float = field(
        default=1e-300,
        metadata={
            "provenance": (
                "Minimum genotype term before logarithms (probability); keeps family likelihood "
                "terms finite."
            ),
        },
    )


@dataclass(frozen=True)
class RateParameters:
    """Genome-wide recombination-rate estimate from relatives and its adoption gate (rl20).

    Measured on 85 simulated runs and the TroMau cross (estimator version 5 of the 4.18 rate study); not user
    settings. Rates are in cM/Mb, likelihood thresholds in nats, probabilities dimensionless.
    """

    site_posterior: float = field(
        default=0.99,
        metadata={
            "provenance": (
                "Minimum leave-pair-out posterior of a family's parental genotype pair for a site to enter a "
                "pair's inheritance chain (probability); the pair's chain conditions on that pair exactly."
            ),
        },
    )
    segregation_posterior: float = field(
        default=0.9,
        metadata={
            "provenance": (
                "Minimum posterior of the same unordered parental pair from the other members alone, both slots "
                "latent (probability); a sequenced slot's confident false heterozygous calls cannot select sites "
                "its children do not segregate at."
            ),
        },
    )
    minimum_information: int = field(
        default=2,
        metadata={
            "provenance": (
                "Minimum family information, members - 2 + 2 x sequenced slots (count); a family below it cannot "
                "determine its parents' genotypes without the pair."
            ),
        },
    )
    rate_low: float = field(
        default=0.2,
        metadata={"provenance": "Lowest rate of the likelihood grid (cM/Mb)."},
    )
    rate_high: float = field(
        default=50.0,
        metadata={"provenance": "Highest rate of the likelihood grid (cM/Mb)."},
    )
    rate_points: int = field(
        default=48,
        metadata={
            "provenance": "Log-spaced rate grid points (count); the maximiser is refined by a parabola on log rate."
        },
    )
    artefacts: tuple[tuple[float, float], ...] = field(
        default=((0.0, 20.0), (0.1, 20.0), (0.1, 200.0), (1.0, 20.0), (1.0, 200.0), (10.0, 20.0), (10.0, 200.0)),
        metadata={
            "provenance": (
                "Artefact-stretch nuisance grid: (start rate per Mb, mean length kb) pairs, fitted with the "
                "rate by maximum likelihood and shared by the data set; simulations select 0-0.1 per Mb, TroMau "
                "0.1-1 per Mb."
            ),
        },
    )
    error_rates: tuple[float, ...] = field(
        default=(0.0, 0.001, 0.003, 0.01, 0.03, 0.1),
        metadata={
            "provenance": (
                "Isolated genotype-error grid (probability per site): each selected site's state emissions are "
                "mixed with their state mean, e' = (1 - eps) e + eps mean(e), and eps is fitted by maximum likelihood "
                "with the rate; errors the genotype likelihoods do not represent otherwise read as inheritance "
                "changes (simulations select 0-0.001, TroMau 0.03, a gain of 712 nats)."
            ),
        },
    )
    inconsistency_nats: float = field(
        default=10.0,
        metadata={
            "provenance": (
                "Family exclusion threshold (nats): a family whose own composite maximum exceeds its value at "
                "the leave-one-family-out maximiser by more than this is dropped, one family at a time; read on "
                "the chi-square(1) scale (p about 1e-5)."
            ),
        },
    )
    bootstrap_replicates: int = field(
        default=1000,
        metadata={"provenance": "Percentile-bootstrap replicates of the interval (count)."},
    )
    bootstrap_seed: int = field(
        default=417,
        metadata={"provenance": "Fixed seed of the bootstrap draws (integer); the interval is reproducible."},
    )
    minimum_interval_units: int = field(
        default=5,
        metadata={
            "provenance": (
                "Minimum equal-material units (1-Morgan units of the working map's support axis, DESIGN_417 D5) for "
                "the two-level bootstrap (families, then each drawn family's units), else minimum families for a "
                "family bootstrap (count); a bootstrap over fewer units is degenerate and the estimate falls back. "
                "Units replace 4.18.1's chromosomes (owner, 13 Nov 2026: the amount of material decides, never a count "
                "of chromosomes); five units are five working Morgans, the pedigree's assessable material."
            ),
        },
    )
    minimum_determined_sites: float = field(
        default=1000.0,
        metadata={
            "provenance": (
                "Minimum meiosis-weighted mean of the families' median determined sites per pair (sites); the "
                "depth and design guard: 1x runs score 2-200, 5x runs 3,000-20,000."
            ),
        },
    )
    maximum_half_width: float = field(
        default=0.25,
        metadata={
            "provenance": (
                "Maximum 95 % interval half-width relative to the estimate (fraction); about 60 expected "
                "crossovers (informative meioses x genome Morgans) reach it."
            ),
        },
    )
    minimum_adoption_families: int = field(
        default=5,
        metadata={
            "provenance": (
                "Minimum informative families for rate_estimate = adopt to adopt the estimate (count); structural: with fewer, "
                "between-family variation in the rate (sex, background) cannot be measured, whatever the interval."
            ),
        },
    )
    edge_low: float = field(
        default=0.21,
        metadata={"provenance": "An estimate at or below this lies at the low grid edge and falls back (cM/Mb)."},
    )
    edge_high: float = field(
        default=49.0,
        metadata={"provenance": "An estimate at or above this lies at the high grid edge and falls back (cM/Mb)."},
    )
    working_digits: int = field(
        default=3,
        metadata={
            "provenance": (
                "Significant digits of an adopted working rate (digits); far below the interval's width, so the "
                "working map's identity does not depend on last-bit arithmetic."
            ),
        },
    )


# Fixed numerical and workspace controls; not user settings.
LOG_ZERO = -1e300  # Log-domain maximum initializer (nats).
SCREEN_FLOOR = 1e-30  # Float32 site-product fallback threshold (probability).
SCALE_EXPONENT = 500  # Power-of-two product-rescaling exponent (binary exponent).
EVIDENCE_COMPARE_CHUNK = 256  # Early-exit evidence-comparison chunk width (values).
SCREEN_SITE_TILE = 256  # Storage-only site tile width (sites).
TERM_FLOOR = 1e-300  # Smallest class-pair site term before its logarithm (probability; rl21).

__all__ = ["AlleleFrequencyParameters", "FamilyParameters", "RateParameters"]
