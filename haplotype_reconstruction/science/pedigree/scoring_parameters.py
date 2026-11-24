"""Pedigree preparation, transmission scoring and candidate panel parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreparationParameters:
    """Marker subsampling of the prepared painting components."""

    # At most this many evenly spaced selected markers per painting bin enter the transmission model.
    markers_per_bin: int = field(
        default=10,
        metadata={
            "provenance": "PS01; select at most 10 evenly spaced markers per painting bin for transmission scoring.",
        },
    )


@dataclass(frozen=True)
class ScoringParameters:
    """Projected transmission model and forward scoring of M0/M1/M2 configurations. Child emissions read the
    children's mixed likelihoods (the painting's outlier rate against the cohort law; science.shared.observation):
    no allele-misread probability of their own (4.21, A8)."""

    # A candidate-parent bridge mixes in the uniform selector kernel with weight
    # switch probability x (1 - min(1, n / N)), n = the smaller informative selected-marker count of
    # the two adjacent parent bins, N = this value; 0 keeps the painter-interpolated chain.
    source_gap_support_sites: int = field(
        default=1,
        metadata={
            "provenance": (
                "PS04; 1 informative selected marker supplies full adjacent-bin support; less support mixes a "
                "uniform selector excursion."
            ),
        },
    )
    # Maximum-entropy bridge: post-check tolerance and independent bisection budget.
    bridge_postcheck_tolerance: float = field(
        default=2e-13,
        metadata={
            "provenance": (
                "PS04; dimensionless base tolerance 2e-13 for bridge probability and marginal residual checks."
            ),
        },
    )
    bridge_max_iterations: int = field(
        default=128,
        metadata={
            "provenance": "PS04; at most 128 bisection iterations for the maximum-entropy bridge root.",
        },
    )
    # Null source of an unobserved parent's slot: it starts from, and redraws from, the founder-state frequencies
    # transmitted by the other samples with an informative painting site (leave the child out), with a uniform
    # pseudo-count of this many samples (keeps a state no other sample carries possible).
    null_prior_samples: float = field(
        default=1.0,
        metadata={
            "provenance": (
                "PS04; 1 uniform pseudo-sample keeps every founder state possible in the leave-child-out null "
                "source."
            ),
        },
    )


@dataclass(frozen=True)
class PanelParameters:
    """Genome-wide M2 panel of bounded retained-parent pairs."""

    # Every eligible pair (p, q) of a child's genome-retained M1 parents enters the M2 panel: there is no relevance
    # screen (DESIGN_417 S8, 14 Nov 2026). A screen's exclusion guarantee covers only the fits it screens, and both
    # screens measured (full data alone; full data plus leave-one-chromosome-out fits) dropped pairs that bootstrap
    # and leave-one-unit-out fits select, changing released calls at low material. At most this many pairs of a
    # child enter (an owner decision): above it, the pairs with the largest full-data bound on their joint M2
    # likelihood (pedigree.pair_bound: both parents' transmission chains against the child's genotypes, bin by bin,
    # summed over the genome). Pairs tied with the last kept one (an equal bound) all enter, and a pair whose bound is
    # not a number is always kept. A child with more pairs is flagged (CandidatePanel.capped_children, the search
    # diagnostics' m2_panel_capped): its M2 evidence then covers fewer pairs, a lower bound, so its calls rest on an
    # incomplete panel. The cap binds when a child retains more than 23 parents (24 give 276 pairs): at low depth and
    # in large sib-rich cohorts. It bounds the panel at cap x N rows, so M2 scoring stays linear in N per child.
    max_pairs_per_child: int = field(
        default=256,
        metadata={
            "provenance": (
                "PS07; nominal cap of 256 M2 pairs per child, every retained eligible pair below it (no relevance "
                "screen, DESIGN_417 S8, owner 14 Nov 2026), the largest full-data bounds above it; boundary ties "
                "and unknown bounds remain, and binding is flagged."
            ),
        },
    )
    # States with at least this transmitted-state mass keep their own group in the pair bound (a parent's other named
    # states are aggregated). Any value keeps the bound valid; it only sets its tightness and cost.
    pair_bound_high_mass: float = field(
        default=0.01,
        metadata={
            "provenance": (
                "PS07; probability-mass cutoff 0.01 for separate named-state groups; controls bound tightness "
                "and cost."
            ),
        },
    )
    # The retention band and the chromosome softmax temperature T = units ** information_tempering_power
    # (m1.temperature), units the chromosome's selected markers with an empty bin as one (the genome's for the genome
    # band): the band keeps every candidate within 60 T nats of the child's best M1, a compute budget with the pair cap.
    information_tempering_power: float = field(
        default=0.5,
        metadata={
            "provenance": (
                "PS07; dimensionless exponent 0.5 on the selected-marker count sets the retention band's and the "
                "chromosome softmax temperature (4.21 B2 at r_T = 0; 4.22 R4: a compute budget of the latent-order "
                "decision, wide band and no cap giving the same decisions on every measured root)."
            ),
        },
    )
    # Chromosome weights sqrt(markers) / median are clipped to [1 / ratio, ratio].
    maximum_contig_weight_ratio: float = field(
        default=4.0,
        metadata={
            "provenance": (
                "PS07; dimensionless ratio 4 clips sqrt(marker count) relative to its median before normalizing "
                "weights."
            ),
        },
    )
    # Utility = rank_weight x tied rank evidence + (1 - rank_weight) x contaminated softmax.
    rank_weight: float = field(
        default=0.35,
        metadata={
            "provenance": (
                "PS07; probability weight 0.35 for tied-rank evidence in diagnostic utility; the remainder uses "
                "contaminated softmax."
            ),
        },
    )
    # Softmax mixed with the uniform distribution over eligible parents at this weight.
    chromosome_contamination: float = field(
        default=0.02,
        metadata={
            "provenance": (
                "PS07; probability weight 0.02 of the uniform eligible-parent distribution in chromosome "
                "diagnostic utility."
            ),
        },
    )
