"""Family-phase model, solver and release constants.

Only ``protected_intervals`` is user-visible; the stage supplies decoded breakpoint intervals.
Recombination rates come from the chromosome's genetic map; the
founder label process comes from the painting's estimated copying model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...core.errors import SettingsError


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
      junction_pin_bins: the flank window of that test (bins, cut at the relative's own label changes).
    Release (release.py):
      site_deviation: mixture mass of the release's site model: given its label pair, a sample's ordered alleles at a
        site are the pair's panel alleles (each with its own leave-one-out flip probability, allele_evidence.py), or
        with this probability redrawn from a uniform genotype (a label error shorter than a bin, a conversion or
        mutation; the genotype then changes with probability 2/3 of it). Weighed by the sample's own reads at the
        site, reads that contradict the labels' genotype lower, withdraw or move the call. A working value chosen on
        the validation simulations (0.01 unphases 12-13% of phased joins; 3e-4 keeps more genotype errors), not an
        estimate of an error rate. The panel alleles' own uncertainty is separate: it is each state's
        capped empirical-Bayes per-copy disagreement with the selected allele, conditional on the fitted carriers
        and panel, with the sample's direct own likelihood term removed. The copies flip independently; this is not
        a cross-fitted refit or exact integration over carrier/panel selection or one shared latent allele.
      genotype_posterior: a genotype is called where its posterior reaches this value (the painting's tau).
      phase_posterior: a heterozygous call is released phased where its ordered allele posterior reaches this
        value.
      set_posterior: a gauge sample's phase set continues while the orientation of its alleles, given the set's
        anchor, stays at or above this value at its heterozygous bins (the median over a bin's confidently
        heterozygous sites of max(P01, P10) / (P01 + P10)); uncertainty about which labels it carries does not
        break a set. Stricter than phase_posterior because a wrong continuation misphases the rest of the set,
        whereas a wrong site call misphases one site. A working value chosen on the validation simulations: at 0.99,
        near-ties decided only by the label-switch prior added released long switches in partial-genome designs,
        which 0.999 avoided. Not an estimate of an error rate, nor a calibrated confidence for a whole set; longer
        sets still carry an upstream identity error over more sites. It must lie below the orientation certainty of a
        fully determined
        heterozygous site under the site model, (1 - 5 site_deviation / 6) / (1 - 2 site_deviation / 3)
        (0.99983 at the default site_deviation), or every bin would end a set.
        Phase sets also break at junction candidates for samples carrying both labels unless a parental homologue
        pins them, and, for samples without a linked parent, at painting tie boundaries (release.py); a sample with
        children keeps a tie break only where its children-anchored orientation certainty at the nearest
        confidently heterozygous bin on either side is below this value. Its odds, log(set_posterior /
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
    junction_pin_bins: int = field(
        default=10,
        metadata={
            "provenance": "FP10; inspect up to 10 bins per flank, stopping at parental label changes.",
            "units": "bins",
        },
    )
    site_deviation: float = field(
        default=1e-3,
        metadata={
            "provenance": (
                "FP16; genotype-departure mixture probability 0.001, a working value chosen on validation "
                "simulations."
            ),
            "units": "probability",
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
            "provenance": "FP17-FP18/FP21; probability cutoff 0.99 for ordered heterozygote and meiosis-chain release.",
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
        for name in ("max_sweeps", "junction_min_samples", "junction_pin_bins"):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise SettingsError(f"family_phase.{name} must be a positive integer")
        for name in ("genotype_posterior", "phase_posterior", "set_posterior"):
            if not 0.5 < getattr(self, name) < 1:
                raise SettingsError(f"family_phase.{name} must lie strictly between 0.5 and 1")
        if self.set_posterior < self.phase_posterior:
            raise SettingsError("family_phase.set_posterior must be at least phase_posterior")
        ceiling = (1 - 5 * self.site_deviation / 6) / (1 - 2 * self.site_deviation / 3)
        refused = not self.set_posterior < ceiling
        if refused:
            raise SettingsError(
                f"family_phase.set_posterior must lie below {ceiling:.6f}, the orientation certainty of a "
                f"fully determined heterozygous site at site_deviation {self.site_deviation}"
            )
        if not 0 < self.site_deviation < 0.5:
            raise SettingsError("family_phase.site_deviation must lie strictly between 0 and 0.5")
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
# Whole-bin site windows of the per-sample release loops: a workspace size (windows are whole bins, so every site
# posterior, call and bin orientation is the same at any size). It is large because each window costs one Python step
# per sample: with 4096-site windows most of 112 release threads waited for the interpreter (chr3, 0.5x).
RELEASE_WINDOW = 1 << 16
EMPTY_COMPONENT_BIN_SITES = 100


__all__ = ["PhaseParameters"]
