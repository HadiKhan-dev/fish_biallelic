"""Family-phase model, solver and release constants.

Only ``protected_intervals`` is user-visible; the stage supplies decoded breakpoint intervals.
Recombination rates come from the chromosome's genetic map; the
founder label process comes from the painting's estimated copying model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.errors import SettingsError


# Independent site blocks bound scoring scratch per worker. The prevalence solver uses
# a fixed double-precision bracket and snaps its numerical endpoints to exact 0 and 1.
SITE_ERROR_BATCH = 384  # sites
SITE_ERROR_BISECTIONS = 54  # iterations to resolve the unit interval in float64
SITE_ERROR_PREVALENCE_BOUNDARY = 1e-15  # probability, within rounding of a bracket endpoint


@dataclass(frozen=True)
class PhaseParameters:
    """Pedigree founder-mosaic phase of one chromosome (docs/methods/family_phase.md).

    Model:
      not_copy_rate_per_bp: rate per bp at which a homologue linked to a parent enters the not-a-copy state (a
        painting or panel error, conversion or mutation, or a false edge); its label is then free.
      not_copy_mean_bp: mean length of a not-a-copy run.
    Solver (exact block coordinate ascent):
      max_sweeps: work cap on block sweeps. Every applied block raises the objective by more than
        ``gain_quantum`` nats, so the ascent terminates without it; a component that reaches the cap keeps its
        (valid) current state and is recorded as not settled.
      gain_quantum: smallest objective gain (nats) for which a block path is applied.
      tie_quantum: Viterbi scores within this many nats are tied; ties keep the current state, then the lowest
        index (the highest under the explicit convention reversal).
    Identity junctions (bin edges where panel rows may have exchanged founder identity):
      junction_min_samples: distinct samples whose free or not-a-copy tracks must switch between the same two
        labels at one bin edge for the edge to be a junction candidate.
      junction_pin_nats: a carrier of both junction labels keeps its phase join across a candidate only when the
        parental homologue it copies there is pinned by the parent's own reads: on each flank the parent's pair beats
        every pair putting the other junction label on that homologue, whatever the partner, by at least this many
        nats.
      junction_pin_reference_bins: the flank window of that test (reference windows, cut at the relative's own label changes).
    Release (release.py):
      The release's site model: given its label pair, a sample's ordered alleles at a site are the pair's panel
        alleles (each with its own leave-one-out flip probability, allele_evidence.py), or, with the run's outlier
        rate (estimated at the painting calibration), redrawn from the site's cohort genotype frequencies (ordered
        weights q0, q1/2, q1/2, q2; science.shared.observation). In 4.20 this was a fixed working value 0.001 against
        a uniform genotype, chosen on the validation simulations (0.01 unphased 12-13% of phased joins; 3e-4 kept more
        genotype errors).
      genotype_posterior: a genotype is called where its posterior reaches this value (the painting's tau).
      phase_posterior: a heterozygous call is eligible for phased release where its ordered allele posterior
        reaches this value. The same threshold defines high propagated noncopy support for the local agreement
        mask, requiring as strong support for uncertain origin as for calling a copy source.
      set_posterior: a gauge sample's phase set continues while the orientation of its alleles, given the set's
        anchor, stays at or above this value at its heterozygous bins (the median over a bin's witness sites, those
        confidently heterozygous under the set statistic's site model that the labels alone do not make confidently
        homozygous, of max(P01, P10) / (P01 + P10), release.py); uncertainty about which labels it carries does not
        break a set. Stricter than
        phase_posterior because a wrong continuation misphases the rest of the set, whereas a wrong site call
        misphases one site. A working value chosen on the validation simulations: at 0.99,
        near-ties decided only by the label-switch prior added released long switches in partial-genome designs,
        which 0.999 avoided. Not an estimate of an error rate, nor a calibrated confidence for a whole set; longer
        sets still carry an upstream identity error over more sites. The orientation certainty of a fully
        determined heterozygous site under the release's site model is (1 - eps + eps q1 / 2) / (1 - eps + eps q1),
        below this value wherever the alternative allele is common (eps 0.0055 on TroMau), so the set statistic uses
        its own site model, set_site_deviation against a uniform genotype, and never the outlier rate.
      set_site_deviation: the set statistic's site departure against a uniform genotype (ordered weights 1/3, 1/6,
        1/6, 1/3), 4.20's working value. It has two jobs: a label pair whose panel alleles cannot produce the reads
        keeps a 50/50 orientation instead of dropping out of the statistic (with no departure, label uncertainty
        vanishes exactly where the reads contradict it, and such near-ties released long switches; a site that the
        labels alone make confidently homozygous is heterozygous only through the departure and is not a witness,
        release.py), and the ceiling
        it gives a fully determined heterozygous site,
        (1 - 5d/6) / (1 - 2d/3) = 0.99983, must lie above set_posterior, so such a site never ends a set by itself,
        while near-ties decided by the label prior still do. With the run's outlier rate instead, every confidently
        heterozygous bin of a gauge sample ended its set on real data (4.21 D3: TroMau sets 40,916 in 4.20 against
        3.1 M; validation replays lab421/r11/f1).
        Phase sets also break at junction candidates for samples carrying both labels unless a parental homologue
        pins them, and, for samples without a linked parent, at painting tie boundaries (release.py); a sample with
        children keeps a tie break only where its children-anchored orientation certainty (the same statistic) at
        the nearest bin with a witness site on either side is below this value. Its odds, log(set_posterior /
        (1 - set_posterior)) = 6.9 nats, also decide a parent's pairing at a double label change (model.py): a joint
        move to the other pairing is kept only when it raises the objective by more than that, and a pairing whose
        alternative is closer breaks the parent's and its children's sets there (a working margin against one
        optimized alternative, not a calibrated pairing posterior).
      protected_intervals: (contig, start, end) closed bp intervals across which meiosis chains reset (the switch
        probability is 1/2): reference-order breakpoints, and both ends of reversed segments.
    """

    not_copy_rate_per_bp: float = field(
        default=1e-8,
        metadata={
            "provenance": "FP02; rate 1e-8 per bp for a linked homologue entering the free not-a-copy state.",
            "units": "per bp",
        },
    )
    not_copy_mean_bp: float = field(
        default=5e4,
        metadata={
            "provenance": "FP02; mean not-a-copy run length 50000 bp sets the exit rate.",
            "units": "bp",
        },
    )
    max_sweeps: int = field(
        default=30,
        metadata={
            "provenance": (
                "FP07-FP08; cap of 30 block-sweep passes; reaching it retains the current valid state as "
                "unsettled."
            ),
            "units": "passes",
        },
    )
    gain_quantum: float = field(
        default=1e-6,
        metadata={
            "provenance": "FP07; strict minimum accepted local objective gain of 1e-6 nats.",
            "units": "nats",
        },
    )
    tie_quantum: float = field(
        default=1e-9,
        metadata={
            "provenance": (
                "FP04; absolute Viterbi tie band 1e-9 nats, preferring the incumbent then the explicit index "
                "convention."
            ),
            "units": "nats",
        },
    )
    junction_min_samples: int = field(
        default=2,
        metadata={
            "provenance": "FP10; require 2 distinct switching samples to nominate a founder-identity junction.",
            "units": "samples",
        },
    )
    junction_pin_nats: float = field(
        default=10.0,
        metadata={
            "provenance": "FP10; require 10 nats of parental own-read evidence per flank to pin a homologue.",
            "units": "nats",
        },
    )
    junction_pin_reference_bins: int = field(
        default=10,
        metadata={
            "provenance": "FP10; inspect up to 10 painting reference windows per flank, stopping at parental label changes.",
            "units": "reference windows",
        },
    )
    genotype_posterior: float = field(
        default=0.9,
        metadata={
            "provenance": "FP17-FP19; probability cutoff 0.9 for genotype calls and informative heterozygosity.",
            "units": "probability",
        },
    )
    phase_posterior: float = field(
        default=0.99,
        metadata={
            "provenance": (
                "FP17-FP21; probability cutoff 0.99 for ordered heterozygotes and meiosis-chain release, also "
                "requiring equivalently strong propagated noncopy support in both solves for a local phase mask."
            ),
            "units": "probability",
        },
    )
    set_posterior: float = field(
        default=0.999,
        metadata={
            "provenance": (
                "FP08/FP18-FP19; probability cutoff 0.999 for phase-set continuation; its log odds also gate "
                "parental pairing moves."
            ),
            "units": "probability",
        },
    )
    set_site_deviation: float = field(
        default=0.001,
        metadata={
            "provenance": (
                "4.20's working site departure against a uniform genotype, kept for the phase-set statistic only "
                "(4.21 D3: the run's outlier rate capped every common heterozygote below set_posterior)."
            ),
            "units": "probability",
        },
    )
    site_error_rates: tuple[float, ...] = field(
        default=(2.0**-10, 2.0**-9, 2.0**-8, 2.0**-7, 2.0**-6, 2.0**-5, 2.0**-4, 2.0**-3),
        metadata={
            "units": "probability",
            "provenance": "D1 round 2; dyadic slab spans sparse to substantial recurrent wrong-allele reads in "
                          "either homozygote, with equal weight per direction and rate.",
        },
    )
    site_error_bic_penalty: float = field(
        default=0.5,
        metadata={
            "units": "nats per log site count",
            "provenance": "D1 round 2; marginal gain must exceed the BIC charge for one fitted contig prevalence.",
        },
    )
    site_error_direction_posterior: float = field(
        default=0.5,
        metadata={
            "units": "probability",
            "provenance": "D1 round 2; use a strict posterior majority for one homozygote after marginalising the "
                          "rate and including the no-excess state.",
        },
    )
    protected_intervals: tuple[tuple[str, int, int], ...] = field(
        default=(),
        metadata={
            "provenance": (
                "FP02; supplied closed reference-order breakpoint intervals in bp; empty means no protected "
                "intervals."
            ),
            "units": "bp",
        },
    )

    def __post_init__(self):
        if not 0 < self.not_copy_rate_per_bp < 1e-3 or not self.not_copy_mean_bp > 0:
            raise SettingsError("family_phase not-a-copy rates must be positive (rate per bp below 1e-3)")
        for name in ("max_sweeps", "junction_min_samples", "junction_pin_reference_bins"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise SettingsError(f"family_phase.{name} must be a positive integer")
        for name in ("genotype_posterior", "phase_posterior", "set_posterior"):
            if not 0.5 < getattr(self, name) < 1:
                raise SettingsError(f"family_phase.{name} must lie strictly between 0.5 and 1")
        if self.set_posterior < self.phase_posterior:
            raise SettingsError("family_phase.set_posterior must be at least phase_posterior")
        if not 0 < self.set_site_deviation < 0.5:
            raise SettingsError("family_phase.set_site_deviation must lie strictly between 0 and 0.5")
        ceiling = (1 - 5 * self.set_site_deviation / 6) / (1 - 2 * self.set_site_deviation / 3)
        if not self.set_posterior < ceiling:
            raise SettingsError(
                f"family_phase.set_posterior must lie below {ceiling:.6f}, the orientation certainty of a fully "
                f"determined heterozygous site at set_site_deviation {self.set_site_deviation}"
            )
        if not self.junction_pin_nats > 0:
            raise SettingsError("family_phase.junction_pin_nats must be positive")
        if not self.gain_quantum > 0 or not self.tie_quantum >= 0:
            raise SettingsError("family_phase quanta must be positive")

    def protected_for(self, contig):
        """Closed bp intervals of ``contig`` where meiosis chains reset (reference-order breakpoints)."""
        return [(int(start), int(end)) for name, start, end in self.protected_intervals if name == contig]


# Fixed probability floors and log-weight guards for transitions and release (FP02, FP14, FP15, FP21).
LABEL_SWITCH_MINIMUM = 1e-15
MEIOSIS_SWITCH_MINIMUM = 1e-12
NOT_COPY_ENTRY_MINIMUM = 1e-12
NOT_COPY_EXIT_MINIMUM = 1e-6
MEIOSIS_RESET_TOLERANCE = 1e-12
FLIP_NEGLIGIBLE = 1e-15
FLIP_CEILING = 0.5
PRIOR_BOUNDS = (0.01, 0.99)
LOGIT_LIMIT = 700.0
POSTERIOR_LOG_FLOOR = -700.0
IMPOSSIBLE_LOG_WEIGHT = -1e300
# FP14/FP22: fixed chunks preserve carrier accumulation and complete-bin medians.
EVIDENCE_CHUNK = 4096
SITE_WINDOW = 4096
# FP20: independent bin batches for the two noncopy scans. Bound each worker's scratch to 512 bins per track.
ORIGIN_BIN_BATCH = 512
# Whole-bin site windows of the per-sample release loops: a workspace size (windows are whole bins, so every site
# posterior, call and bin orientation is the same at any size). It is large because each window costs one Python step
# per sample: with 4096-site windows most of 112 release threads waited for the interpreter (chr3, 0.5x).
RELEASE_WINDOW = 1 << 16
# Forward steps of an anchored gauge release are taken in pieces of about this many transition terms (bins x block
# states x predecessor states), and the continuation test is applied after each piece: a workspace size (the released
# values do not depend on it). One piece is one Python step of the sample's thread, so it is kept large enough that
# release threads do not queue for the interpreter, and small enough that a phase set that ends early wastes little.
RELEASE_ADVANCE_TERMS = 1 << 22
EMPTY_COMPONENT_BIN_SITES = 100


__all__ = ["PhaseParameters"]
